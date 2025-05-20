import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from scripts.sentiment_analysis import set_llm_client, classify_sentiment
from scripts.config import LABEL_MAP

def evaluate_model(model_name, val_data, num_samples=None):
    set_llm_client(model_name)
    y_true = []
    y_pred = []
    if num_samples:
        val_data = val_data.select(range(num_samples))
    for example in tqdm(val_data, desc=f"Evaluating {model_name}"):
        tweet = example["text"]
        true_label = LABEL_MAP[example["label"]]
        pred = classify_sentiment(tweet)
        y_true.append(true_label)
        y_pred.append(pred)
    acc = accuracy_score(y_true, y_pred)
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {acc:.3f}")
    print("Classification Report:\n", classification_report(y_true, y_pred, labels=["Bearish", "Bullish", "Neutral"]))
    return y_true, y_pred

if __name__ == "__main__":
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
    val_data = dataset["validation"]

    for model in ["HuggingFaceH4/zephyr-7b-beta", "openai/gpt-3.5-turbo"]:
        evaluate_model(model, val_data, num_samples=200)
    # models = ["HuggingFaceH4/zephyr-7b-beta", "openai/gpt-3.5-turbo"]
    # all_results = {}
    
    # for model in models:
    #     all_results[model] = evaluate_model(model, val_data)
    
    # # Comparative analysis
    # comparison = pd.DataFrame({
    #     "Model": models,
    #     "Accuracy": [all_results[m]["accuracy"] for m in models],
    #     "Weighted F1": [all_results[m]["class_metrics"]["weighted avg"]["f1-score"] for m in models]
    # })
    # print("\nModel Comparison:")
    # print(comparison)