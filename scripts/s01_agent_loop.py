# 原仓库地址: https://github.com/zht7063/learn-cc-openai

import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import subprocess

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSC"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

SYS_PROMPT = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def agent_loop(messages: list):
    while True:
        resp_msg = (
            client.chat.completions.create(
                model=os.getenv("DEFAULT_MODEL"),
                messages=messages,
                tools=TOOLS,
            )
            .choices[0]
            .message
        )

        # 将 llm 的回复添加到 messages 中
        messages.append({
            "role": "assistant",
            "content": resp_msg.content,
            "tool_calls": resp_msg.tool_calls,
        })

        # 如果模型没有尝试调用工具，结束本轮 agent-loop。
        if not resp_msg.tool_calls:
            return

        # 执行每个 tool call 并收集结果
        for tc in resp_msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"\033[33m$ {args['command']}\033[0m")
            output = run_bash(args["command"])
            print(output[:200])
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })

if __name__ == "__main__":
    history = [{"role": "system", "content": SYS_PROMPT}]

    while True:
        try:
            query = input("ur query:> ")
        except (EOFError, KeyboardInterrupt):
            break
        
        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})

        agent_loop(history)

        response_content = history[-1]["content"]
        print(response_content)
        print()
