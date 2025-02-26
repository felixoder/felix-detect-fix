import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

detector_name = "./bug_detector_model"
fixer_name = "./bug_fixer_model"

# Automatically select the best available device (GPU > MPS > CPU)
device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)

# Use FP16 if on GPU, else FP32
torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(detector_name)
model = AutoModelForSequenceClassification.from_pretrained(detector_name, torch_dtype=torch_dtype).to(device)

fixer_tokenizer = AutoTokenizer.from_pretrained(fixer_name)
fixer_model = AutoModelForCausalLM.from_pretrained(
    fixer_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)

def classify_code(code):
    inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return "bug-free" if predicted_label == 0 else "buggy"

def fix_buggy_code(code):
    prompt = f"### Fix this buggy Python code:\n{code}\n### Fixed Python code:\n"
    inputs = fixer_tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = fixer_model.generate(**inputs, max_length=256, do_sample=False, num_return_sequences=1)

    fixed_code = fixer_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return fixed_code.split("### Fixed Python code:")[1].strip() if "### Fixed Python code:" in fixed_code else fixed_code

if __name__ == "__main__":
    command = sys.argv[1]
    code = sys.argv[2]

    if command == "classify":
        print(classify_code(code))
    elif command == "fix":
        print(fix_buggy_code(code))
