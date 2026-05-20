"""
演示使用 Chat Completions API + requests 实现支持工具调用的命令行连续对话。

相比于上一个版本，增加了 Function Calling 能力，不依赖 OpenAI SDK，
直接通过 requests 发送 HTTP 请求实现工具调用循环。
"""

import json
import os
import urllib3
from datetime import datetime

import requests
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML

# ===========================
# 配置部分
# ===========================
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"

PROXIES = {
    "http": "http://127.0.0.1:13128",
    "https": "http://127.0.0.1:13128",
}
VERIFY_SSL = False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===========================
# HTTP 会话（复用连接）
# ===========================
session = requests.Session()
session.proxies = PROXIES
session.verify = VERIFY_SSL

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ===========================
# 工具定义（JSON Schema 格式）
# ===========================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "计算两个数字之和",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个加数"},
                    "b": {"type": "number", "description": "第二个加数"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply_numbers",
            "description": "计算两个数字之积",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个乘数"},
                    "b": {"type": "number", "description": "第二个乘数"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取本地当前时间",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "时区名称，默认 local",
                        "default": "local",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，支持中英文，例如 Beijing 或 北京",
                    },
                },
                "required": ["city"],
            },
        },
    },
]


# ===========================
# 工具实现
# ===========================
def add_numbers(a: float, b: float, **_) -> str:
    return str(a + b)


def multiply_numbers(a: float, b: float, **_) -> str:
    return str(a * b)


def get_current_time(timezone: str = "local", **_) -> str:
    if timezone != "local":
        return f"当前示例仅支持 local 时区，你传入的是: {timezone}"
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_weather(city: str, **_) -> str:
    # Demo 固定返回值，不调用真实 API
    return (
        f"{city} 当前天气：晴转多云，"
        f"气温 22°C（体感 20°C），"
        f"湿度 60%，风速 15 km/h"
    )


TOOL_HANDLERS = {
    "add_numbers": add_numbers,
    "multiply_numbers": multiply_numbers,
    "get_current_time": get_current_time,
    "get_weather": get_weather,
}


def execute_tool(name: str, arguments_json: str) -> str:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return f"工具不存在: {name}"
    try:
        args = json.loads(arguments_json or "{}")
        return handler(**args)
    except Exception as exc:
        return f"工具执行失败: {exc}"


# ===========================
# 聊天记录
# ===========================
conversation_history = [
    {
        "role": "system",
        "content": (
            "你是一个数学专家，但是说话非常毒舌，喜欢用尖刻的语言回答问题。"
            "请使用中文回答。"
            "当问题涉及计算、时间或天气时，优先调用可用工具。"
        ),
    }
]


# ===========================
# 工具调用循环 + 流式输出最终回复
# ===========================
def get_response_with_tools(user_input: str) -> str:
    conversation_history.append({"role": "user", "content": user_input})

    try:
        # --- 工具调用循环（非流式，便于解析 tool_calls）---
        while True:
            data = {
                "model": MODEL,
                "messages": conversation_history,
                "tools": TOOLS,
                "tool_choice": "auto",
                "max_tokens": 1024,
                "temperature": 1.0,
                "stream": False,
            }
            resp = session.post(API_URL, headers=HEADERS, data=json.dumps(data))
            resp.raise_for_status()
            result = resp.json()

            message = result["choices"][0]["message"]
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                # 无工具调用 → 得到最终文本，流式重新请求以保留流式体验
                break

            # 将含 tool_calls 的 assistant 消息存入历史
            conversation_history.append(message)

            for call in tool_calls:
                name = call["function"]["name"]
                arguments = call["function"]["arguments"]
                print(f"\033[90m[tool] 调用 {name} 参数: {arguments}\033[0m")
                tool_result = execute_tool(name, arguments)
                print(f"\033[90m[tool] 结果: {tool_result}\033[0m")

                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps({"result": tool_result}, ensure_ascii=False),
                })

        # --- 流式输出最终回复 ---
        data = {
            "model": MODEL,
            "messages": conversation_history,
            "tools": TOOLS,
            "tool_choice": "none",  # 已完成工具调用，禁止再次触发
            "max_tokens": 1024,
            "temperature": 1.0,
            "stream": True,
        }
        print("\033[91mAI:\033[0m ", end="", flush=True)
        full_reply = []
        with session.post(API_URL, headers=HEADERS, data=json.dumps(data), stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    line = line[len("data: "):]
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        print(delta, end="", flush=True)
                        full_reply.append(delta)
                except json.JSONDecodeError:
                    continue
        print()

        reply = "".join(full_reply)
        conversation_history.append({"role": "assistant", "content": reply})
        return reply

    except Exception as e:
        print("请求出错:", e)
        return ""


# ===========================
# 命令行交互
# ===========================
def main():
    print("=== 欢迎使用支持工具调用的连续对话 CLI ===")
    print("输入 'exit' 退出程序\n")

    while True:
        user_input = prompt(HTML("<ansired>你:</ansired> ")).strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break

        get_response_with_tools(user_input)
        print()


if __name__ == "__main__":
    main()
