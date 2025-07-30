FROM python:3.11-slim

WORKDIR /app

# 安装 uv
RUN pip install uv

# 复制项目文件
COPY pyproject.toml README.md ./
COPY . .

# 使用 uv 安装依赖
RUN uv pip install -e .

EXPOSE 8000

CMD ["python", "vision_mcp_server.py"]