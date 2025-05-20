import os
from openai import OpenAI

from huggingface_hub import InferenceClient
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from scripts.config import LABEL_MAP, HF_API_TOKEN, OPENAPI_API_TOKEN

openapi_client = OpenAI(api_key=OPENAPI_API_TOKEN)
llm_client = None
local_pipeline = None
selected_model = None

def set_llm_client(model_name):
    global llm_client, local_pipeline, selected_model, openapi_client
    selected_model = model_name

    if model_name.startswith("local/"):
        model_path = model_name.replace("local/", "")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        local_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
        llm_client = None
    elif model_name.startswith("openai/"):
        llm_client = None
        local_pipeline = None
    else:
        llm_client = InferenceClient(model=model_name, token=HF_API_TOKEN)
        local_pipeline = None

def generate_completion(prompt: str, max_tokens=30):
    global llm_client, local_pipeline, selected_model, openapi_client

    if local_pipeline:
        if "Tweet:" in prompt:
            text = prompt.split("Tweet:")[-1].split("Sentiment:")[0].strip()
        else:
            text = prompt.strip()
        result = local_pipeline(text)[0]
        return result["label"]

    elif selected_model and selected_model.startswith("openai/"):
        try:
            response = openapi_client.chat.completions.create(model=selected_model.replace("openai/", ""),
            messages=[
                {"role": "system", "content": "You are a financial sentiment classifier."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0)
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[OpenAI Error: {e}]"

    elif llm_client:
        try:
            return llm_client.text_generation(prompt, max_new_tokens=max_tokens).strip()
        except Exception as e:
            return f"[HF Error: {e}]"

    return "[Model not initialized]"

def classify_sentiment(tweet):
    prompt = f"Classify the sentiment of this financial tweet as Bearish, Bullish, or Neutral:\n\nTweet: {tweet}\nSentiment:"
    response = generate_completion(prompt, max_tokens=5)
    for label in LABEL_MAP.values():
        if label.lower() in response.lower():
            return label
    return "Unknown"