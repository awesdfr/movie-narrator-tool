# 电影解说视频制作工具 Dockerfile
# 多阶段构建，优化镜像大小

# ==================== 构建阶段 ====================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# 复制前端依赖文件
COPY frontend/package*.json ./

# 安装依赖
RUN npm ci

# 复制前端源码
COPY frontend/ ./

# 构建前端（输出到 backend/static）
RUN npm run build

# ==================== 生产阶段 ====================
FROM python:3.10-slim AS production

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 创建必要的目录
RUN mkdir -p /app/videos/movies \
    /app/videos/narrations \
    /app/videos/reference_audio \
    /app/videos/subtitles \
    /app/models \
    /app/temp \
    /app/temp/logs \
    /app/projects

# 复制后端依赖
COPY backend/requirements.txt ./

# 安装 Python 依赖
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir scipy librosa webrtcvad-wheels typing && \
    pip install --no-cache-dir resemblyzer==0.1.4 --no-deps && \
    pip install --no-cache-dir -r requirements.txt

# 复制后端代码
COPY backend/ ./

# 从前端构建阶段复制静态文件
COPY --from=frontend-builder /app/backend/static ./static

# 复制启动脚本
COPY <<'EOF' /app/start.sh
#!/bin/bash
set -e

echo "========================================"
echo "  电影解说视频制作工具"
echo "  Movie Narrator Tool"
echo "========================================"

# 确保目录存在
mkdir -p /app/videos/movies /app/videos/narrations /app/videos/reference_audio /app/videos/subtitles
mkdir -p /app/models /app/temp /app/projects

# 检查健康状态
echo "检查服务健康状态..."

# 启动应用
echo "启动服务..."
exec python main.py
EOF

RUN chmod +x /app/start.sh

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["/app/start.sh"]

# ==================== 开发阶段 ====================
FROM production AS development

# 开发环境额外安装调试工具
RUN pip install --no-cache-dir pytest pytest-asyncio httpx debugpy

# 开发模式启动
ENV DEBUG=true

CMD ["python", "main.py"]
