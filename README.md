# Financial RAG Demo

面向银行业监管制度与统计报表问答的可解释、确定性 RAG 在线演示。本目录是独立的 Render Docker Web Service 部署副本，不修改生产工作树或 RAG 核心算法。

## Runtime

- Docker / Python 3.12 / FastAPI / Uvicorn
- CPU-only
- LLM：Disabled
- 无 GPU、外部 API Key 或在线模型下载要求
- SQLite 随镜像打包并作为只读检索索引使用

## Public endpoints

- Demo：`/?demo=1`
- Root：`/`（307 跳转到 `/?demo=1`）
- Health：`/health`

`/health.git_commit` 返回 Render 注入的完整 `RENDER_GIT_COMMIT`，用于确认公网实例与部署仓库 commit 一致。

服务监听 `0.0.0.0`，优先使用 Render 注入的 `PORT`，本地缺省端口为 `7860`。

## Local equivalent test

使用 Render 风格环境变量和实际入口启动：

```powershell
$env:PYTHONPATH = "src"
$env:PORT = "7860"
$env:COMPETITION_PUBLIC_MODE = "1"
python -m reg_rag.web_app
```

在另一个终端运行：

```powershell
python verify_deployment.py
```

## Docker

```bash
docker build -t financial-rag-render .
docker run --rm -p 7860:7860 -e PORT=7860 financial-rag-render
```

随后访问 `http://0.0.0.0:7860/?demo=1`，或运行 `python verify_deployment.py`。

## Render

仓库根目录包含 `render.yaml` 和 `Dockerfile`。在 Render 中连接此 GitHub 仓库并创建 Docker Web Service；Build Command 与 Start Command 均留空，由 Dockerfile 完成构建和启动。无需添加额外环境变量、数据库、Redis、持久磁盘、Cron Job 或 Worker。

Render Free Web Service 在空闲时可能休眠，首次访问可能需要等待冷启动。

## Security

公开仓库不得加入 `.env`、凭据、API Key、Token、私有语料、个人路径或运行时上传文件。当前公开模式禁用上传、索引重建、文档删除、LLM 与内部调试接口。
