"""使用 OpenAI SDK + pydantic_function_tool 实现支持工具调用的命令行连续对话。"""

import json
import os
from datetime import datetime
from typing import Any, cast

import httpx
from openai import OpenAI, pydantic_function_tool
from openai.types.chat import ChatCompletionMessageParam
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from pydantic import BaseModel, Field

# ===========================
# 配置部分
# ===========================
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"

# 代理设置(通过自定义httpx客户端实现，支持HTTP或SOCKS5)，并且忽略 SSL 验证
_http_client = httpx.Client(
    proxy="http://127.0.0.1:13128",
    verify=False,
)

# ===========================
# OpenAI 客户端（含代理，忽略 SSL 验证）
# ===========================
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    http_client=_http_client,
)


# ===========================
# 定义工具参数（Pydantic）
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


TOOLS = [
    pydantic_function_tool(
        AddArgs,
        name="add_numbers",
        description="计算两个数字之和",
    ),
    pydantic_function_tool(
        MultiplyArgs,
        name="multiply_numbers",
        description="计算两个数字之积",
    ),
    pydantic_function_tool(
        CurrentTimeArgs,
        name="get_current_time",
        description="获取本地当前时间",
    ),
    pydantic_function_tool(
        GetWeatherArgs,
        name="get_weather",
        description="获取指定城市的当前天气信息",
    ),
]

TOOL_MODELS = {
    "add_numbers": AddArgs,
    "multiply_numbers": MultiplyArgs,
    "get_current_time": CurrentTimeArgs,
    "get_weather": GetWeatherArgs,
}


# ===========================
# 聊天记录
# ===========================
conversation_history: list[dict[str, Any]] = [
    {
        "role": "system",
        "content": (
            "你是一个数学专家，但是说话非常毒舌，喜欢用尖刻的语言回答问题。"
            "请使用中文回答。"
            "当问题涉及计算、时间或天气时，优先调用可用工具。"
        ),
    }
]


def execute_tool_call(tool_call: Any) -> str:
    """执行一次工具调用并返回字符串结果。"""
    function_obj = getattr(tool_call, "function", None)
    if function_obj is None:
        return "工具调用格式不支持：缺少 function 字段"

    tool_name = getattr(function_obj, "name", "")
    raw_args = getattr(function_obj, "arguments", "{}") or "{}"

    model_cls = TOOL_MODELS.get(tool_name)
    if model_cls is None:
        return f"工具不存在: {tool_name}"

    try:
        parsed_args = model_cls.model_validate_json(raw_args)
    except Exception as exc:
        return f"工具参数解析失败: {exc}; 原始参数: {raw_args}"

    try:
        return parsed_args.execute()
    except Exception as exc:
        return f"工具执行失败: {exc}"


def get_response_with_tools(user_input: str) -> str:
    """追加用户消息后，让模型自动选择是否调用工具，直到返回最终文本回复。"""
    try:
        conversation_history.append({"role": "user", "content": user_input})

        while True:
            response = client.chat.completions.create(
                model=MODEL,
                messages=cast(list[ChatCompletionMessageParam], conversation_history),
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=1024,
                temperature=1.0,
            )

            message = response.choices[0].message
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
            }

            if message.tool_calls:
                serialized_tool_calls: list[dict[str, Any]] = []
                for call in message.tool_calls:
                    function_obj = getattr(call, "function", None)
                    if function_obj is None:
                        continue
                    serialized_tool_calls.append(
                        {
                            "id": call.id,
                            "type": call.type,
                            "function": {
                                "name": getattr(function_obj, "name", ""),
                                "arguments": getattr(function_obj, "arguments", "{}"),
                            },
                        }
                    )
                assistant_message["tool_calls"] = serialized_tool_calls
                conversation_history.append(assistant_message)

                for tool_call in message.tool_calls:
                    function_obj = getattr(tool_call, "function", None)
                    if function_obj is None:
                        continue

                    tool_name = getattr(function_obj, "name", "")
                    tool_args = getattr(function_obj, "arguments", "{}")
                    print(f"\033[90m[tool] 调用 {tool_name} 参数: {tool_args}\033[0m")
                    tool_result = execute_tool_call(tool_call)
                    print(f"\033[90m[tool] 结果: {tool_result}\033[0m")

                    conversation_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps({"result": tool_result}, ensure_ascii=False),
                        }
                    )
                continue

            conversation_history.append(assistant_message)
            final_text = message.content or ""
            print("\033[91mAI:\033[0m", final_text)
            return final_text
    except Exception as exc:
        print("请求出错:", exc)
        return ""


# ===========================
# 命令行交互
# ===========================
def main() -> None:
    print("=== 欢迎使用连续对话 CLI（支持工具调用）===")
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
