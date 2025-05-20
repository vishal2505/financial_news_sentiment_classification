from scripts.config import LABEL_MAP
from scripts.sentiment_analysis import set_llm_client, classify_sentiment
from eval.perplexity_eval import compute_perplexity
from eval.llm_judge_eval import llm_judge
from huggingface_hub import InferenceClient
from scripts.config import LABEL_MAP, HF_API_TOKEN


selected_model = "HuggingFaceH4/zephyr-7b-beta"
example = {
    "text": "$ICPT - FDA OKs Intercept’s NDA for obeticholic acid; shares up 5% premarket https://t.co/VgfLhFDoWm",
    "label": 1
}

tweet = example["text"]
label = LABEL_MAP[example["label"]]

llm_client = InferenceClient(model=selected_model, token=HF_API_TOKEN)
print(llm_client)

pred = llm_client.text_generation(tweet, max_new_tokens=5).strip()

print(pred)