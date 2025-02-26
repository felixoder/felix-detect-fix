# Felix Detect & Fix - VS Code Extension

![Felix Detect & Fix](https://img.shields.io/badge/VS%20Code-Extension-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A powerful VS Code extension that detects and fixes code bugs using **machine learning**. This extension integrates with a **bug detection model** and a **bug-fixing model** hosted on Hugging Face, allowing developers to improve code quality efficiently.

## Features ✨
- 🚀 **Detect Bugs**: Classifies code as "buggy" or "bug-free."
- 🔧 **Fix Bugs**: Automatically suggests fixes for detected issues.
- 💡 **Manual Control**: Users decide when to run the detection and fixing functions.
- ⚡ **Fast & Local Processing**: Uses Hugging Face models **locally**, avoiding API calls.

## Installation 🛠️
1. Download and install the extension from the [VS Code Marketplace](https://marketplace.visualstudio.com/vscode).
2. Ensure you have **Node.js** and **VS Code** installed.
3. Open VS Code and enable the extension.

## Usage 🚀
1. **Detect Bugs**:
   - Open a code file.
   - Run the command: `Felix: Detect Bugs`
   - The extension highlights buggy code sections.

2. **Fix Bugs**:
   - After detecting bugs, run `Felix: Fix Bugs`
   - The model suggests code fixes.

## Installation from Source 🏗️
1. Clone the repository:
   ```sh
   git clone https://github.com/felixoder/felix-detect-fix.git
   cd felix-detect-fix
   ```
2. Install dependencies:
   ```sh
   npm install
   ```
3. Package the extension:
   ```sh
   vsce package
   ```
4. Install the packaged `.vsix` file in VS Code.

## Requirements 📦
- VS Code **1.70+**
- Node.js **18+**
- Hugging Face models:
  - [Bug Detector](https://huggingface.co/felixoder/bug_detector_model)
  - [Bug Fixer](https://huggingface.co/felixoder/bug_fixer_model)

## Contributing 🤝
1. Fork the repo & create a new branch.
2. Make your changes & commit.
3. Open a Pull Request!

## License 📜
This project is licensed under the MIT License.

---
Made with ❤️ by [Debayan Ghosh](https://github.com/felixoder).


