"""openai sdk也不再用了，使用langchain的ChatOpenAI封装来实现流式命令行对话。"""


import os
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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
#
# 许多LLM提供商的API都兼容OpenAI接口规范，因此我们可以使用LangChain的ChatOpenAI来调用DeepSeek的Chat Completions API。
# 详见：https://docs.langchain.com/oss/python/concepts/providers-and-models#openai-compatible-endpoints
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
# 聊天记录
# ===========================
conversation_history = []
conversation_history.append(
    SystemMessage(content="你是一个数学专家，但是说话非常毒舌，喜欢用尖刻的语言回答问题。请使用中文回答。")
)


# ===========================
# 获取模型回复（流式输出）
# ===========================
def get_response():
    try:
        print("\033[91mAI:\033[0m ", end="", flush=True)
        full_reply = []
        for chunk in llm.stream(conversation_history): # 这里使用stream方法获取流式输出；invoke是同步调用，会阻塞直到完整回复返回
            delta = chunk.content or ""
            if delta:
                print(delta, end="", flush=True)
                full_reply.append(delta)
        print()  # 换行
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
        # user_input 判空检查
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break

        conversation_history.append(HumanMessage(content=user_input))
        reply = get_response()
        conversation_history.append(AIMessage(content=reply))
        print()


if __name__ == "__main__":
    main()