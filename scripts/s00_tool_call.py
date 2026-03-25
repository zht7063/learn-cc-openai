import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSC"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


# ── 本地工具实现 ──────────────────────────────────────────────
def add(a: int, b: int) -> int:
    return a + b


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "计算两个整数的和",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "第一个加数"},
                    "b": {"type": "integer", "description": "第二个加数"},
                },
                "required": ["a", "b"],
            },
        },
    }
]

TOOL_MAP = {"add": add}

# ── 第一轮：带 tools 发送用户消息 ────────────────────────────
messages = [{"role": "user", "content": "请计算 15 加 27 等于多少？"}]

response = client.chat.completions.create(
    model=os.getenv("DEFAULT_MODEL"),
    messages=messages,
    tools=TOOLS,
)

assistant_msg = response.choices[0].message

if not assistant_msg.tool_calls:
    # 模型未发起工具调用，直接输出（兼容端点可能不支持 tool_calls）
    print("模型未调用工具，直接回复：", assistant_msg.content)
    raise SystemExit

# ── 执行工具调用并构造回传消息 ────────────────────────────────
messages.append(assistant_msg)

for tc in assistant_msg.tool_calls:
    fn = TOOL_MAP[tc.function.name]
    args = json.loads(tc.function.arguments)
    result = fn(**args)
    print(f"[tool] {tc.function.name}({args}) => {result}")
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

# ── 第二轮：将工具结果交回模型，获取自然语言回复 ──────────────
final = client.chat.completions.create(
    model=os.getenv("DEFAULT_MODEL"),
    messages=messages,
)

print(final.choices[0].message.content)
