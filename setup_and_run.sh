#!/bin/bash

# The fair minumum installation script if you want you can modify this according to your usage.

set -e # Exit on error

# Ensure huggingface-cli is installed
if ! command -v huggingface-cli &>/dev/null; then
    echo "Installing huggingface-cli..."
    pip install --upgrade huggingface_hub
fi

# Create directories and download models
echo "Downloading models using huggingface-cli..."
huggingface-cli download felixoder/bug_detector_model --local-dir ./bug_detector_model --repo-type model
huggingface-cli download felixoder/bug_fixer_model --local-dir ./bug_fixer_model --repo-type model

# installing the requirement modules

pip install torch
pip install transformers
pip install 'accelerate>=0.26.0'

echo "Models are downloaded successfully."

# Create the Python script
echo "Creating run_model.py..."
cat >run_model.py <<'EOF'
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
    if len(sys.argv) < 3:
        print("Usage: python3 run_model.py [classify|fix] \"<code>\"")
        sys.exit(1)

    command = sys.argv[1]
    code = sys.argv[2]

    if command == "classify":
        print(classify_code(code))
    elif command == "fix":
        print(fix_buggy_code(code))
EOF

echo "run_model.py created successfully."

# Make run_model.py executable
chmod +x run_model.py

echo "Setup complete. You can now use:"


echo "  $PYTHON_CMD run_model.py classify \"print('Hello World')\""
echo "  $PYTHON_CMD run_model.py fix \"print(Hello World)\""
