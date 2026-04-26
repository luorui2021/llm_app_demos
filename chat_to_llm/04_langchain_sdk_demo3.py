"""
使用 langchain.agents.create_agent 构建带工具调用的chat agent。
相比于上一版本，这个版本的 agent 可以不仅可以自动判断何时需要调用工具，还能自动选择调用哪个
工具，并且可能会自动调用多次工具，直至达到目的或者用尽工具调用次数限制为止。
"""

import os
import datetime
from typing import Any
import httpx
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from pydantic import SecretStr

# ===========================
# 配置部分
# ===========================
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your_api_key_here")
BASE_URL = "https://api.deepseek.com/v1"

# 代理设置（支持HTTP或SOCKS5）
PROXY_URL = "http://127.0.0.1:13128"

# ===========================
# LangChain ChatOpenAI 客户端（含代理，忽略 SSL 验证）
# ===========================
llm = ChatOpenAI(
    api_key=SecretStr(API_KEY),
    base_url=BASE_URL,
    model="deepseek-chat",
    max_completion_tokens=1024,
    temperature=1.0,
    streaming=True,
    http_client=httpx.Client(
        proxy=PROXY_URL,
        verify=False,
    ),
)

# ===========================
# 工具定义
# ===========================
@tool
def get_current_time() -> str:
    """获取当前的日期和时间"""
    print(f"\033[33m[工具执行: get_current_time()]\033[0m")
    return datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气情况

    Args:
        city: 城市名称，例如：北京、上海
    """
    # Demo：固定返回天气数据
    print(f"\033[33m[工具执行: get_weather({city})]\033[0m")
    return f"{city}：晴，气温25°C，湿度60%，微风"


tools = [get_current_time, get_weather]
SYSTEM_PROMPT = "你是一个毒舌智能助手，喜欢嘲讽人，但也很有用。请使用中文回答。"
agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)

# ===========================
# 聊天记录
# ===========================
conversation_history = []


# ===========================
# 获取模型回复（agent 自动处理工具调用循环，并流式输出）
# ===========================
def extract_text_from_chunk(message_chunk: Any) -> str:
    """从 LangChain/LangGraph 流式事件里的消息块中提取可打印文本。

    `agent.stream(..., stream_mode="messages")` 返回的消息块类型并不完全固定，
    在不同模型、不同版本的 LangChain/LangGraph 下，可能出现以下几种常见形态：

    1. 直接就是字符串
    2. 具有 `text` 属性的消息块对象
    3. 具有 `content` 属性，且 `content` 本身是字符串
    4. 具有 `content` 属性，且 `content` 是由多个内容片段组成的列表

    这个函数的目标很单纯：尽量从这些可能的结构里抽取出“当前这一次增量输出
    对应的纯文本”，供终端逐段打印。若当前块中没有可显示文本，则返回空字符串。
    """

    # 某些实现会直接把增量文本作为 str 返回，此时可以直接输出。
    if isinstance(message_chunk, str):
        return message_chunk

    # 有些消息块对象会暴露 text 属性；如果它已经是非空字符串，优先使用。
    # 这样可以少做一层 content 解析。
    text_value = getattr(message_chunk, "text", None)
    if isinstance(text_value, str) and text_value:
        return text_value

    # 更常见的情况是消息块对象把内容放在 content 里。
    # content 既可能是完整字符串，也可能是分片结构。
    content = getattr(message_chunk, "content", "")

    # 如果 content 已经是字符串，直接返回即可。
    if isinstance(content, str):
        return content

    # 某些模型返回的 content 是一个列表，列表中每个元素代表一个内容片段。
    # 这里只提取 type == "text" 的片段，并按原顺序拼接成最终可打印文本。
    # 其他非文本片段（例如工具调用、结构化块等）会被忽略。
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )

    # 兜底：遇到当前版本未覆盖的 chunk 结构时，不抛异常，直接返回空字符串，
    # 这样流式输出流程仍可继续执行。
    return ""


def get_response():
    try:
        print("\033[91mAI:\033[0m ", end="", flush=True)
        full_reply = []

        for stream_event in agent.stream(
            {"messages": conversation_history},
            stream_mode="messages",
        ):
            if not isinstance(stream_event, tuple) or len(stream_event) != 2:
                # 非预期的事件格式，直接忽略
                continue

            message_chunk, metadata = stream_event
            if not isinstance(metadata, dict) or metadata.get("langgraph_node") != "model":
                # 只处理模型输出的消息块，忽略工具调用等其他事件
                continue

            delta = extract_text_from_chunk(message_chunk)
            if delta:
                print(delta, end="", flush=True)
                full_reply.append(delta)

        print()
        return "".join(full_reply)
    except Exception as e:
        print("请求出错:", e)
        return ""


# ===========================
# 命令行交互
# ===========================
def main():
    print("=== 欢迎使用连续对话 CLI ===")
    print("输入 'exit' 退出程序\n")

    while True:
        user_input = prompt(HTML("<ansired>你:</ansired> ")).strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break

        conversation_history.append(HumanMessage(content=user_input))
        reply = get_response()
        if reply:
            conversation_history.append(AIMessage(content=reply))
        print()


if __name__ == "__main__":
    main()
