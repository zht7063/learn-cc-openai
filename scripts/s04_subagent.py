# https://github.com/shareAI-lab/learn-claude-code/blob/main/agents/s04_subagent.py

import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import subprocess
from pathlib import Path

load_dotenv()

WORKDIR = Path.cwd()
client = OpenAI(
    api_key=os.getenv("DASHSC"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

SYSTEM = f"You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks."
SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."

# -- Tool implementations shared by parent and child --


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "【错误】危险命令被阻止"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# Child gets all base tools except task (no recursive spawning)
CHILD_TOOLS = [
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
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
]


# -- Subagent: fresh context, filtered tools, summary-only return --
def run_subagent(prompt: str) -> str:
    sub_messages = [
        {"role": "system", "content": SUBAGENT_SYSTEM},  # system prompt
        {"role": "user", "content": prompt},
    ]  # fresh context

    for _ in range(30):  # max 30 rounds
        response = (
            client.chat.completions.create(
                model=os.getenv("DEFAULT_MODEL"),
                messages=sub_messages,
                tools=CHILD_TOOLS,
                max_tokens=12000,
            )
            .choices[0]
            .messaege
        )

    sub_messages.append(
        {
            "role": "assistant",
            "content": response.content,
        }
    )


# -- Subagent: OpenAI SDK implementation --
def run_subagent_openai(prompt: str) -> str:
    sub_messages = [{"role": "system", "content": SUBAGENT_SYSTEM}]
    sub_messages.append({"role": "user", "content": prompt})

    for _ in range(30):  # safety limit
        response = client.chat.completions.create(
            model=os.getenv("DEFAULT_MODEL"),
            messages=sub_messages,
            tools=CHILD_TOOLS,
            max_tokens=8000,
        )

        choice = response.choices[0]
        assistant_msg = choice.message
        sub_messages.append(
            {
                "role": "assistant",
                "content": assistant_msg.content,
                "tool_calls": assistant_msg.tool_calls,
            }
        )

        if choice.finish_reason != "tool_calls":
            break

        # Handle tool calls
        tool_results = []
        for tool_call in assistant_msg.tool_calls:
            handler = TOOL_HANDLERS.get(tool_call.function.name)
            args = json.loads(tool_call.function.arguments)
            output = (
                handler(**args)
                if handler
                else f"Unknown tool: {tool_call.function.name}"
            )
            tool_results.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": str(output)[:50000],
                }
            )
        sub_messages.append({"role": "tool", "content": tool_results})

    # Only the final text returns to the parent -- child context is discarded
    final_content = choice.message.content
    return final_content if final_content else "(no summary)"


# -- Parent tools: base tools + task dispatcher --
PARENT_TOOLS = CHILD_TOOLS + [
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Task prompt for the subagent",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short description of the task",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
]


def agent_loop(messages: list):
    while True:
        response = client.chat.completions.create(
            model=os.getenv("DEFAULT_MODEL"),
            messages=messages,
            tools=PARENT_TOOLS,
            max_tokens=12000,
        )
        choice = response.choices[0]
        assistant_msg = choice.message

        messages.append(
            {
                "role": assistant_msg.role,
                "content": assistant_msg.content,
                "tool_calls": assistant_msg.tool_calls,
            }
        )

        if choice.finish_reason != "tool_calls":
            return

        results = []
        for tc in assistant_msg.tool_calls:
            handler = TOOL_HANDLERS.get(tc.function.name)
            args = json.loads(tc.function.arguments)
            output = handler(**args) if handler else f"Unknown tool: {tc.function.name}"
            results.append(
                {
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "content": str(output)[:50000],
                }
            )
        messages.append({"role": "tool", "content": results})


if __name__ == "__main__":
    messages = [
        {
            "role": "system",
            "content": SYSTEM,
        }
    ]

    while True:
        try:
            query = input("\033[36ms04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        messages.append({"role": "user", "content": query})
        agent_loop(messages)

        response_content = messages[-1]["content"]
        print(response_content)
        print()
