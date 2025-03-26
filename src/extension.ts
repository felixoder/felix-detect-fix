import * as vscode from "vscode";

import { exec } from "child_process";

function runPythonModel(
  command: "classify" | "fix",
  code: string,
): Promise<string> {
  return new Promise((resolve, reject) => {
    exec(
      `python3 run_model.py ${command} "${code.replace(/"/g, '\\"')}"`,
      (error, stdout, stderr) => {
        if (error) {
          reject(`Error: ${stderr}`);
        } else {
          resolve(stdout.trim());
        }
      },
    );
  });
}

export function activate(context: vscode.ExtensionContext) {
  let detectBugCommand = vscode.commands.registerCommand(
    "bugFixer.detectBug",
    async () => {
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
          vscode.window
            .showWarningMessage("Buggy Code Detected", "Fix Bug")
            .then((selection) => {
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
    },
  );

  let fixBugCommand = vscode.commands.registerCommand(
    "bugFixer.fixBug",
    async () => {
      const editor = vscode.window.activeTextEditor;

      if (!editor) {
        vscode.window.showErrorMessage("No active editor found.");

        return;
      }

      const code = editor.document.getText();

      vscode.window.showInformationMessage("Fixing bug...");

      try {
        const fixedCode = await runPythonModel("fix", code);

        editor.edit((editBuilder) => {
          const fullRange = new vscode.Range(
            editor.document.positionAt(0),

            editor.document.positionAt(code.length),
          );

          editBuilder.replace(fullRange, fixedCode);
        });

        vscode.window.showInformationMessage("Bug fixed!");
      } catch (error: any) {
        vscode.window.showErrorMessage(error);
      }
    },
  );

  context.subscriptions.push(detectBugCommand, fixBugCommand);
}

export function deactivate() { }
