import streamlit as st
from datasets import Dataset, load_dataset
import random
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.config import LABEL_MAP
from scripts.sentiment_analysis import set_llm_client, classify_sentiment
from eval.perplexity_eval import compute_perplexity
from eval.llm_judge_eval import llm_judge

dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
val_data = dataset["validation"]

def select_random_diverse_examples(dataset, label_column="label", num_per_class=1, total=5):
    label_to_examples = {label: [] for label in set(dataset[label_column])}
    for example in dataset:
        label_to_examples[example[label_column]].append(example)
    sampled = []
    for label, examples in label_to_examples.items():
        sampled.extend(random.sample(examples, min(num_per_class, len(examples))))
    remaining = total - len(sampled)
    if remaining > 0:
        remaining_pool = [ex for ex in dataset if ex not in sampled]
        sampled.extend(random.sample(remaining_pool, min(remaining, len(remaining_pool))))
    return Dataset.from_list(sampled)

st.set_page_config(page_title="LLM Sentiment Analyzer", layout="wide")
st.title("Financial Tweet Sentiment Classifier with LLM Evaluation")

# Model selection
available_models = [
    "HuggingFaceH4/zephyr-7b-beta",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "openai/gpt-3.5-turbo",
    "openai/gpt-4",
    "local/custom_sentiment_model"
]
selected_model = st.selectbox("Choose a model for prediction:", available_models)
set_llm_client(selected_model)

# Session state
if "results" not in st.session_state:
    st.session_state.results = []

# Mode selection
mode = st.radio("Choose input mode:", ["Use validation set sample", "Enter custom financial news"])

if mode == "Use validation set sample":
    # Prediction from validation set
    if st.button("🔍 Predict Sentiments from Validation Set"):
        st.session_state.results = []
        st.subheader("Predicted Sentiments")
        diverse_val_samples = select_random_diverse_examples(val_data, total=5)
        for example in diverse_val_samples:
        #for example in val_data.select(range(5)):
            tweet = example["text"]
            label = LABEL_MAP[example["label"]]
            pred = classify_sentiment(tweet)
            st.session_state.results.append({
                "tweet": tweet,
                "human": label,
                "pred": pred
            })
            st.markdown(f"**Tweet:** {tweet}\n\n**Predicted:** {pred}\n\n**Actual:** {label}")

elif mode == "Enter custom financial news":
    st.subheader("Enter Your Own Financial News Text")
    user_input = st.text_area("Input text:", height=100)
    actual_label = st.selectbox("(Optional) Select actual label for evaluation", ["Bearish", "Bullish", "Neutral", "Unknown"])

    if st.button("🔍 Predict Custom Sentiment") and user_input:
        st.session_state.results = []
        pred = classify_sentiment(user_input)
        st.session_state.results.append({
            "tweet": user_input,
            "human": actual_label,
            "pred": pred
        })
        st.markdown(f"**Tweet:** {user_input}\n\n**Predicted:** {pred}\n\n**Actual:** {actual_label}")

# Evaluation section (works for both modes)
if st.session_state.results:
    st.subheader("Evaluate LLM Predictions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Evaluate Perplexity"):
            st.subheader("Perplexity Scores for All Labels")
            for r in st.session_state.results:
                prompt = f"Classify the sentiment of this financial tweet as Bearish, Bullish, or Neutral:\n\nTweet: {r['tweet']}\nSentiment:"
                perplexities = {}
                for label in ["Bearish", "Bullish", "Neutral"]:
                    perplexities[label] = compute_perplexity(prompt, label)

                sorted_ppl = sorted(perplexities.items(), key=lambda x: x[1])

                st.markdown(f"**Tweet:** {r['tweet']}\n\n**Actual:** {r['human']}\n\n**Predicted:** {r['pred']}")
                for label, ppl in sorted_ppl:
                    st.markdown(f"- {label}: Perplexity = {ppl:.2f}")
                st.markdown("---")

    with col2:
        if st.button("Evaluate with LLM-as-a-Judge"):
            st.subheader("LLM-as-a-Judge Verdicts")
            for r in st.session_state.results:
                judgment = llm_judge(r["tweet"], r["human"], r["pred"])
                st.markdown(f"**Tweet:** {r['tweet']}\n\n**Human Label:** {r['human']}\n\n**AI Label:** {r['pred']}\n\n**Judge Verdict:** {judgment}")

    # Accuracy and Confusion Matrix
    if st.button("Show Accuracy & Confusion Matrix"):
        st.subheader("📊 Classification Report")
        y_true = [r["human"] for r in st.session_state.results if r["human"] != "Unknown"]
        y_pred = [r["pred"] for r in st.session_state.results if r["human"] != "Unknown"]

        if y_true and y_pred:
            acc = accuracy_score(y_true, y_pred)
            st.markdown(f"**Accuracy:** {acc * 100:.2f}%")

            st.text("Classification Report:")
            st.text(classification_report(y_true, y_pred, labels=["Bearish", "Bullish", "Neutral"]))

            cm = confusion_matrix(y_true, y_pred, labels=["Bearish", "Bullish", "Neutral"])
            df_cm = pd.DataFrame(cm, index=["Bearish", "Bullish", "Neutral"], columns=["Bearish", "Bullish", "Neutral"])
            st.write("Confusion Matrix:")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig, use_container_width=False)