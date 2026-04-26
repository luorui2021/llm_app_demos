"""
相比于上一版本，这个版本使用了 LangChain 的工具调用能力，演示了如何在对话过程中调用外部工具
（获取当前时间和天气信息）。用户输入后，模型可以根据需要调用工具获取信息，并将结果返回给用户，
实现更智能的对话体验。

这个版本的核心亮点在于：
1. 工具定义：使用 @tool 装饰器定义了两个工具函数，分别用于获取当前时间和天气信息。
2. 模型绑定工具：通过 llm.bind_tools(tools) 将工具绑定到语言模型，使其能够在生成回复时调用
工具。
3. 工具调用循环：在 get_response 函数中实现了一个循环，模型每次生成回复后检查是否有工具调用，
如果有则执行工具并将结果加入对话历史，然后继续生成下一轮回复，直到没有工具调用为止。
4. 流式输出：模型回复仍然是流式输出，用户可以实时看到模型的回复内容和工具调用结果，提升交互体验。
"""

import os
import datetime
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
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
TOOLS_MAP = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

# ===========================
# 聊天记录
# ===========================
conversation_history = []
conversation_history.append(
    SystemMessage(content="你是一个智能助手，可以帮助用户查询时间和天气。请使用中文回答。")
)


# ===========================
# 获取模型回复（支持工具调用循环）
# ===========================
def get_response():
    try:
        while True:
            response = llm_with_tools.invoke(conversation_history)

            # 没有工具调用，直接输出最终回复
            if not response.tool_calls:
                print("\033[91mAI:\033[0m ", response.content)
                return response.content

            # 有工具调用，执行工具并将结果加入历史
            conversation_history.append(response)

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                result = TOOLS_MAP[tool_name].invoke(tool_args)

                conversation_history.append(
                    ToolMessage(content=str(result), tool_call_id=tool_id)
                )
            # 继续循环，将工具结果发回模型
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
