"""演示使用原始Completions API实现最基础的文本补全"""

import os
import requests
import json
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML

# ===========================
# 配置部分
# ===========================
API_KEY = os.environ.get("OPENAI_API_KEY")
API_URL = "https://api.openai.com/v1/completions"

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


# ===========================
# 获取模型回复
# ===========================
def get_response():
    # 将历史对话拼接成完整 prompt
    full_prompt = "".join(conversation_history)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-3.5-turbo-instruct",  # 或其他 Completion 模型： gpt-4o-mini
        "prompt": full_prompt,
        "max_tokens": 20,
        "temperature": 0,
        "stop": None
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
        text = result["choices"][0]["text"].strip()
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
        user_input = prompt(HTML("<ansired>你:</ansired> ")).strip() # 等价于 input("\033[91m你:\033[0m ").strip()
        # user_input 判空检查
        if not user_input:
            user_input = ""
        if user_input.lower() == "exit":
            break

        conversation_history.append(f"{user_input}")
        reply = get_response()
        conversation_history.append(f"{reply}")
        print(f"\033[91mAI:\033[0m {reply}\n")


if __name__ == "__main__":
    main()