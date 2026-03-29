# https://github.com/shareAI-lab/learn-claude-code/blob/main/agents/s06_context_compact.py

import os
import time
import json
import subprocess
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

load_dotenv()

WORKDIR = Path.cwd()
client = OpenAI(
    api_key=os.getenv("DASHSC"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

THRESHOLD = 50000  # 上下文窗口大小
TRANSCRIPT_DIR = WORKDIR / ".transcripts"  # 对话记录目录
KEEP_RECENT = 3  # 保留最近的对话记录数量


def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 chars per token."""
    return len(str(messages)) // 4


# -- Layer1: micro_compact - replace old tool results with placeholders. --
def micro_compact(messages: list) -> list:
    entries = []
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            entries.append((msg.get("tool_call_id", ""), msg))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "tool_call_id" in part:
                    entries.append((part.get("tool_call_id", ""), part))
    if len(entries) <= KEEP_RECENT:
        return messages

    tool_name_map = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tcs = msg.get("tool_calls")
        if not tcs:
            continue
        for tc in tcs:
            if hasattr(tc, "id"):
                tid, name = tc.id, tc.function.name
            else:
                tid = tc["id"]
                name = tc["function"]["name"]
            tool_name_map[tid] = name

    for tid, target in entries[:-KEEP_RECENT]:
        c = target.get("content")
        if isinstance(c, str) and len(c) > 100:
            tool_name = tool_name_map.get(tid, "unknown")
            target["content"] = f"[Previous: used {tool_name}]"
    return messages


# -- Layer2: auto_compact - save transcript, summarize, replace messages.
def auto_compact(messages: list) -> list:
    # save full transcript to disk.
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")

    # Ask LLM to summarize.
    conversation_text = json.dumps(messages, default=str)[:80000]
    response = client.chat.completions.create(
        model=os.getenv("DEFAULT_MODEL"),
        messages=[
            {
                "role": "user",
                "content": "Summarize this conversation for continuity. Include: "
                "1) What was accomplished, 2) Current state, 3) Key decisions made. "
                "Be concise but preserve critical details.\n\n" + conversation_text,
            }
        ],
        max_tokens=2000,
    )
    summary = response.choices[0].message.content

    # Replace old messages with compressed summary.
    return [
        {
            "role": "user",
            "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}",
        },
        {
            "role": "assistant",
            "content": "Understood. I have the context from the summary. Continuing.",
        },
    ]


# -- Tool implementations --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
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
    "compact": lambda **kw: "Manual compression requested.",
}

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
    {
        "type": "function",
        "function": {
            "name": "compact",
            "description": "Trigger manual conversation compression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "What to preserve in the summary",
                    }
                },
            },
        },
    },
]


def agent_loop(messages: list):
    while True:
        # Layer 1: micro_compact before each LLM call.
        micro_compact(messages)
        # Layer 2: auto_compact if token estimate exceeds threshold.
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)
        response = client.chat.completions.create(
            model=os.getenv("DEFAULT_MODEL"),
            messages=messages,
            tools=TOOLS,
            max_tokens=12000,
        )
        choice = response.choices[0]
        assistant_msg = choice.message
        messages.append({
            "role": assistant_msg.role,
            "content": assistant_msg.content,
            "tool_calls": assistant_msg.tool_calls,
        })
        if choice.finish_reason != "tool_calls":
            return

        manual_compact = False
        for tc in assistant_msg.tool_calls:
            name = tc.function.name
            if name == "compact":
                manual_compact = True
                output = "Compressing..."
            else:
                handler = TOOL_HANDLERS.get(name)
                try:
                    args = json.loads(tc.function.arguments)
                    output = (
                        handler(**args) if handler else f"Unknown tool: {name}"
                    )
                except Exception as e:
                    output = f"Error: {e}"
            print(f"> {name}: {str(output)[:200]}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(output)[:50000],
            })

        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)


if __name__ == "__main__":
    messages = [
        {
            "role": "system",
            "content": SYSTEM,
        }
    ]

    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        messages.append({"role": "user", "content": query})
        agent_loop(messages)

        response_content = messages[-1]["content"]
        print(response_content)
        print()