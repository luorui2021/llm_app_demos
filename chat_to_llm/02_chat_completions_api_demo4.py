"""
演示使用 Chat Completions API + requests 实现支持工具调用的命令行连续对话。

相比于 demo3，使用 pydantic_function_tool 自动生成工具 JSON Schema，
简化了 TOOLS 的定义；HTTP 请求仍通过 requests 发送，不使用 OpenAI 客户端。
"""

import json
import os
import urllib3
from datetime import datetime

import requests
from openai import pydantic_function_tool
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from pydantic import BaseModel, Field

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
# 工具定义（Pydantic 模型 + execute 方法）
# ===========================
class AddArgs(BaseModel):
    a: float = Field(description="第一个加数")
    b: float = Field(description="第二个加数")

    def execute(self) -> str:
        return str(self.a + self.b)


class MultiplyArgs(BaseModel):
    a: float = Field(description="第一个乘数")
    b: float = Field(description="第二个乘数")

    def execute(self) -> str:
        return str(self.a * self.b)


class CurrentTimeArgs(BaseModel):
    timezone: str = Field(default="local", description="时区名称，默认 local")

    def execute(self) -> str:
        if self.timezone != "local":
            return f"当前示例仅支持 local 时区，你传入的是: {self.timezone}"
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class GetWeatherArgs(BaseModel):
    city: str = Field(description="城市名称，支持中英文，例如 Beijing 或 北京")

    def execute(self) -> str:
        # Demo 固定返回值，不调用真实 API
        return (
            f"{self.city} 当前天气：晴转多云，"
            f"气温 22°C（体感 20°C），"
            f"湿度 60%，风速 15 km/h"
        )


# pydantic_function_tool 自动从模型生成 JSON Schema，省去手写嵌套 dict
TOOLS = [
    pydantic_function_tool(AddArgs, name="add_numbers", description="计算两个数字之和"),
    pydantic_function_tool(MultiplyArgs, name="multiply_numbers", description="计算两个数字之积"),
    pydantic_function_tool(CurrentTimeArgs, name="get_current_time", description="获取本地当前时间"),
    pydantic_function_tool(GetWeatherArgs, name="get_weather", description="获取指定城市的当前天气信息"),
]

TOOL_MODELS = {
    "add_numbers": AddArgs,
    "multiply_numbers": MultiplyArgs,
    "get_current_time": CurrentTimeArgs,
    "get_weather": GetWeatherArgs,
}


def execute_tool(name: str, arguments_json: str) -> str:
    model_cls = TOOL_MODELS.get(name)
    if model_cls is None:
        return f"工具不存在: {name}"
    try:
        return model_cls.model_validate_json(arguments_json or "{}").execute()
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
            message = resp.json()["choices"][0]["message"]
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                break

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
            "tool_choice": "none",
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
