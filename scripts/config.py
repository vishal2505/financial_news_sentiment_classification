import os
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
OPENAPI_API_TOKEN = os.getenv("OPENAI_API_KEY")
LABEL_MAP = {0: "Bearish", 1: "Bullish", 2: "Neutral"}