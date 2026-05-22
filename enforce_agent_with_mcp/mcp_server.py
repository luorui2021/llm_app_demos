"""
Demo MCP Server — 使用 FastMCP 提供4个样例工具，通过 Streamable HTTP 协议暴露。

启动方式：
    uv run python enforce_agent_with_mcp/mcp_server.py

服务地址：http://127.0.0.1:8000/mcp
"""

from datetime import datetime
from mcp.server.fastmcp import FastMCP

# ===========================
# 服务实例
# ===========================
mcp = FastMCP("Demo Server", host="127.0.0.1", port=8000)


# ===========================
# 工具定义
# ===========================
@mcp.tool()
def add_numbers(a: float, b: float) -> str:
    """计算两个数字之和"""
    return str(a + b)


@mcp.tool()
def multiply_numbers(a: float, b: float) -> str:
    """计算两个数字之积"""
    return str(a * b)


@mcp.tool()
def get_current_time(timezone: str = "local") -> str:
    """获取本地当前时间。timezone 参数目前仅支持 local。"""
    if timezone != "local":
        return f"当前示例仅支持 local 时区，你传入的是: {timezone}"
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@mcp.tool()
def get_weather(city: str) -> str:
    """获取指定城市的当前天气信息（Demo 固定返回值，不调用真实 API）。
    city: 城市名称，支持中英文，例如 Beijing 或 北京。
    """
    return (
        f"{city} 当前天气：晴转多云，"
        f"气温 32°C（体感 30°C），"
        f"湿度 60%，风速 15 km/h"
    )


# ===========================
# 启动 HTTP Server
# ===========================
if __name__ == "__main__":
    print("启动 MCP Server → http://127.0.0.1:8000/mcp")
    mcp.run(transport="streamable-http")
