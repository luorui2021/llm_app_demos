"""
演示使用Chat Completions API实现命令行连续对话。

Chat Completions API已成为行业主流的对话式接口规范，OpenAI之外的主流模型提供商都支持这一接口。
相比于传统的文本补全接口，Chat Completions API原生支持多轮对话上下文，并且在消息结构上更清晰
（区分系统、用户、助手角色），非常适合构建聊天机器人、智能助手等应用。

本示例展示了如何使用原始的HTTP请求来调用Chat Completions API，并实现一个简单的命令行连续对
话界面。用户输入问题后，程序将调用API获取模型回复，并将对话历史保存在内存中，以便在后续的交互
中提供上下文支持。示例中还演示了如何配置代理和忽略SSL验证，以适应不同的网络环境。
"""

import os
import requests
import json
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML

# ===========================
# 配置部分
# ===========================
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
API_URL = "https://api.deepseek.com/v1/chat/completions"

# 代理设置（支持HTTP或SOCKS5）
PROXIES = {
    "http": "http://127.0.0.1:13128",
    "https": "http://127.0.0.1:13128",
}
# 忽略 SSL 验证
VERIFY_SSL = False
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===========================
# 聊天记录
# ===========================
conversation_history = []
conversation_history.append({
    "role": "system",
    "content": "你是一个毒舌智能助手，喜欢用尖刻的语言回答问题。请使用中文回答。"
    })


# ===========================
# 获取模型回复
# ===========================
def get_response():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-chat",
        "messages": conversation_history,
        "max_tokens": 1024,
        "temperature": 1.0,
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            data=json.dumps(data),
            proxies=PROXIES,
            verify=VERIFY_SSL  # 忽略 SSL 验证
        )
        response.raise_for_status()
        result = response.json()
        text = result["choices"][0]["message"]["content"].strip()

        return text
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

        conversation_history.append({"role": "user", "content": user_input})
        reply = get_response()
        conversation_history.append({"role": "assistant", "content": reply})
        print(f"\033[91mAI:\033[0m {reply}\n")


if __name__ == "__main__":
    main()