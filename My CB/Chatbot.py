import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

with open("about_me.txt", "r", encoding="utf-8") as f:
    ABOUT_ME = f.read()

SYSTEM_PROMPT = f"""
You are a personal chatbot.
Answer ONLY using the information below.
If the answer is not present, say you don't know.

Information:
{ABOUT_ME}
"""

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    response = client.chat.completions.create(
        model="deepseek/deepseek-r1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
    )

    print("Bot:", response.choices[0].message.content)
