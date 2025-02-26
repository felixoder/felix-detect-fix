import * as vscode from "vscode";
import { exec } from "child_process";
import * as fs from "fs";
import * as https from "https";
import * as path from "path";

const scriptUrl = "https://raw.githubusercontent.com/felixoder/felix-detect-fix/master/run_model.py";
const modelBaseUrl = "https://huggingface.co/felixoder"; // Change if needed

const scriptPath = path.join(__dirname, "run_model.py");
const detectorPath = path.join(__dirname, "bug_detector_model");
const fixerPath = path.join(__dirname, "bug_fixer_model");

// Function to download a file
function downloadFile(url: string, destination: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (fs.existsSync(destination)) {
      console.log(`File already exists: ${destination}`);
      resolve();
      return;
    }

    console.log(`Downloading ${url}...`);
    const fileStream = fs.createWriteStream(destination);
    https.get(url, (response) => {
      response.pipe(fileStream);
      fileStream.on("finish", () => {
        fileStream.close();
        console.log(`Download complete: ${destination}`);
        resolve();
      });
    }).on("error", reject);
  });
}

// Download missing models and script
async function ensureDependencies() {
  await downloadFile(scriptUrl, scriptPath);
  await downloadFile(`${modelBaseUrl}/bug_detector_model/resolve/main/model.safetensors`, detectorPath);
  await downloadFile(`${modelBaseUrl}/bug_fixer_model/resolve/main/model.safetensors`, fixerPath);
}

// Function to execute the Python model
async function runPythonModel(command: string, code: string): Promise<string> {
  await ensureDependencies(); // Ensure models & script are downloaded

  return new Promise((resolve, reject) => {
    const safeCode = JSON.stringify(code); // Escape user input
    exec(`python3 ${scriptPath} ${command} ${safeCode}`, (error, stdout, stderr) => {
      if (error) {
        reject(`Error: ${error.message}`);
        return;
      }
      if (stderr) {
        reject(`stderr: ${stderr}`);
        return;
      }
      resolve(stdout.trim());
    });
  });
}

// VS Code Extension Activation
export function activate(context: vscode.ExtensionContext) {
  let detectBugCommand = vscode.commands.registerCommand("bugFixer.detectBug", async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showErrorMessage("No active editor found.");
      return;
    }

    const code = editor.document.getText();
    vscode.window.showInformationMessage("Detecting bugs...");

    try {
      const result = await runPythonModel("classify", code);
      if (result === "buggy") {
        vscode.window.showWarningMessage("Buggy Code Detected", "Fix Bug").then(selection => {
          if (selection === "Fix Bug") {
            vscode.commands.executeCommand("bugFixer.fixBug");
          }
        });
      } else {
        vscode.window.showInformationMessage("Code is bug-free!");
      }
    } catch (error: any) {
      vscode.window.showErrorMessage(error);
    }
  });

  let fixBugCommand = vscode.commands.registerCommand("bugFixer.fixBug", async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showErrorMessage("No active editor found.");
      return;
    }

    const code = editor.document.getText();
    vscode.window.showInformationMessage("Fixing bug...");

    try {
      const fixedCode = await runPythonModel("fix", code);
      editor.edit(editBuilder => {
        const fullRange = new vscode.Range(
          editor.document.positionAt(0),
          editor.document.positionAt(code.length)
        );
        editBuilder.replace(fullRange, fixedCode);
      });

      vscode.window.showInformationMessage("Bug fixed!");
    } catch (error: any) {
      vscode.window.showErrorMessage(error);
    }
  });

  context.subscriptions.push(detectBugCommand, fixBugCommand);
}

export function deactivate() {}

