# llm_app_demos

渐进式演示项目，展示从原始 HTTP 调用到远程 MCP 工具调用的 LLM 集成方式。

## 目录结构

```
chat_to_llm/          # 单文件 CLI demo，按技术层级编号（01~04）
enforce_agents_with_mcp/
    mcp_server.py     # FastMCP Streamable HTTP Server（端口 8000）
    agent.py          # Agent：raw requests + mcp SDK + ToolManager
```

## 快速开始

```bash
# 安装依赖
uv sync

# 运行单文件 demo（示例）
uv run python chat_to_llm/04_langchain_sdk_demo3.py

# 运行 Remote MCP Agent demo（需两个终端）
uv run python enforce_agents_with_mcp/mcp_server.py   # 终端1
uv run python enforce_agents_with_mcp/agent.py        # 终端2
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 |

本地代理 `http://127.0.0.1:13128` 已硬编码在各 demo 配置块中（SSL 验证关闭）。