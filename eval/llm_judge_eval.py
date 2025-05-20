from openai import OpenAI
from scripts.config import OPENAPI_API_TOKEN

def build_few_shot_messages(tweet, human_label, ai_label):
    examples = [
        {
            "tweet": "The stock plummeted after poor earnings report.",
            "human": "Bearish",
            "ai": "Bullish",
            "verdict": "No",
            "justification": "The tweet clearly conveys negative market reaction, so 'Bullish' is incorrect."
        },
        {
            "tweet": "Company X is expected to post record profits next quarter.",
            "human": "Bullish",
            "ai": "Bullish",
            "verdict": "Yes",
            "justification": "The AI label matches the optimistic tone of the tweet."
        }
    ]

    messages = [{"role": "system", 
                 "content": "You are a financial sentiment analysis expert. Judge whether the AI-predicted sentiment label matches the actual sentiment of the tweet, and explain your decision."}]

    for ex in examples:
        messages.append({
            "role": "user",
            "content": f"""Tweet: {ex['tweet']}
Human label: {ex['human']}
AI label: {ex['ai']}
Is the AI label correct?"""
        })
        messages.append({
            "role": "assistant",
            "content": f"{ex['verdict']}\nJustification: {ex['justification']}"
        })

    # Add the real input
    messages.append({
        "role": "user",
        "content": f"""Tweet: {tweet}
Human label: {human_label}
AI label: {ai_label}
Is the AI label correct?"""
    })

    return messages

def llm_judge(tweet, human_label, ai_label):
    openapi_client = OpenAI(api_key=OPENAPI_API_TOKEN)
    messages = build_few_shot_messages(tweet, human_label, ai_label)
    try:
        response = openapi_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"[OpenAI Error: {e}]"