from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

def compute_perplexity(prompt, completion):
    input_text = prompt + " " + completion
    encodings = tokenizer(input_text, return_tensors="pt")
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()
