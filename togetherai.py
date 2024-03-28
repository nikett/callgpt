from openai import OpenAI
import os

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")

client = OpenAI(
  api_key=TOGETHER_API_KEY,
  base_url='https://api.together.xyz/v1',
)

chat_completion = client.chat.completions.create(
  messages=[
    {
      "role": "system",
      "content": "You are an expert travel guide.",
    },
    {
      "role": "user",
      "content": "Tell me fun things to do in San Francisco.",
    }
  ],
  model="mistralai/Mixtral-8x7B-Instruct-v0.1"
)

print(chat_completion.choices[0].message.content)