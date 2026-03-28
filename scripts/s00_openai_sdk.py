import os
from openai import OpenAI
from dotenv import load_dotenv
from rich import print

load_dotenv()

# 构建 OpenAI 客户端
client = OpenAI(
    api_key=os.getenv("DASHSC"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# 发送请求
response = client.chat.completions.create(
    model=os.getenv("DEFAULT_MODEL"),
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

# 打印
print(f"response:\n{response}")
print("===== ===== ===== =====")
print(f"response.choices[0].message:\n{response.choices[0].message}")
