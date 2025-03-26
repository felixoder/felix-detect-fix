import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from huggingface_hub import snapshot_download  # Add this import

# Get absolute paths relative to THIS file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Model configuration
MODELS = {
    "detector": {
        "repo": "felixoder/bug_detector_model",
        "path": os.path.join(MODEL_DIR, "detector")
    },
    "fixer": {
        "repo": "felixoder/bug_fixer_model",
        "path": os.path.join(MODEL_DIR, "fixer")
    }
}

# Download models if missing
for model in MODELS.values():
    if not os.path.exists(model["path"]):
        print(f"Downloading {model['repo']}...")
        snapshot_download(
            repo_id=model["repo"],
            local_dir=model["path"],
            local_dir_use_symlinks=False
        )

# Now load the models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

# Load detector model
detector_tokenizer = AutoTokenizer.from_pretrained(
    MODELS["detector"]["path"],
    local_files_only=True
)
detector_model = AutoModelForSequenceClassification.from_pretrained(
    MODELS["detector"]["path"],
    local_files_only=True,
    torch_dtype=torch_dtype
).to(device)

# Load fixer model
fixer_tokenizer = AutoTokenizer.from_pretrained(
    MODELS["fixer"]["path"],
    local_files_only=True
)
fixer_model = AutoModelForCausalLM.from_pretrained(
    MODELS["fixer"]["path"],
    local_files_only=True,
    torch_dtype=torch_dtype
).to(device)

# Rest of your existing functions remain the same...

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
