# Felix Detect & Fix - VS Code Extension

![Felix Detect & Fix](https://img.shields.io/badge/VS%20Code-Extension-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A powerful VS Code extension that detects and fixes code bugs using **machine learning**. This extension integrates with a **bug detection model** and a **bug-fixing model** hosted on Hugging Face, allowing developers to improve code quality efficiently.

## Features ✨

- 🚀 **Detect Bugs**: Classifies code as "buggy" or "bug-free."
- 🔧 **Fix Bugs**: Automatically suggests fixes for detected issues.
- 💡 **Manual Control**: Users decide when to run the detection and fixing functions.
- ⚡ **Fast & Local Processing**: Uses Hugging Face models **locally**, avoiding API calls.
- 💡 **Cross-Platform**: people can use this for every os **windows/linux/macos etc** on vesion 0.0.2 onwards.

## Installation 🛠️

1. Download and install the extension from the [VS Code Marketplace](https://marketplace.visualstudio.com/vscode).
2. Ensure you have **Node.js** and **VS Code** installed.
3. Open VS Code and enable the extension.

## Usage 🚀

````sh
project structure
   |_ your_code.py
   |_ bug*detector*model [download from ```huggingface-cli download felixoder/bug_detector_model --local-dir ./bug_detector_model\n```]
   |_ bug*fixer_model [download from ```huggingface-cli download felixoder/bug_fixer_model --local-dir ./bug_fixer_model```]
   |_ run_model.py [see ## run_model.py]


````

## run_model.py script:

```sh
pip install torch
pip install transformers
```

```sh
import sys

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

detector_name = "./bug_detector_model"
fixer_name = "./bug_fixer_model"

# Automatically select the best available device (GPU > MPS > CPU)
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# Use FP16 if on GPU, else FP32
torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(detector_name)
model = AutoModelForSequenceClassification.from_pretrained(
    detector_name, torch_dtype=torch_dtype
).to(device)

fixer_tokenizer = AutoTokenizer.from_pretrained(fixer_name)
fixer_model = AutoModelForCausalLM.from_pretrained(
    fixer_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)


def classify_code(code):
    inputs = tokenizer(
        code, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return "bug-free" if predicted_label == 0 else "buggy"


def fix_buggy_code(code):
    prompt = f"### Fix this buggy Python code:\n{code}\n### Fixed Python code:\n"
    inputs = fixer_tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = fixer_model.generate(
            **inputs, max_length=256, do_sample=False, num_return_sequences=1
        )

    fixed_code = fixer_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return (
        fixed_code.split("### Fixed Python code:")[1].strip()
        if "### Fixed Python code:" in fixed_code
        else fixed_code
    )


if __name__ == "__main__":
    command = sys.argv[1]
    code = sys.argv[2]

    if command == "classify":
        print(classify_code(code))
    elif command == "fix":
        print(fix_buggy_code(code))


```

2. **Detect Bugs**:

   - Open a python code file.
   - Run the command: `Detect Bugs`
   - The extension highlights buggy code sections.

3. **Use a build template**:

   - Paste this in your terminal

   ```sh
    wget -O setup_and_run.sh https://raw.githubusercontent.com/felixoder/felix-detect-fix/master/setup_and_run.sh
   ```

   ```sh
       chmod +x setup_and_run.sh
   ```

   ```sh
       ./setup_and_run.sh

   ```

4. **Fix Bugs**:
   - After detecting bugs, run `Fix Bugs`
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
