FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖（用于 matplotlib）
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY backend/requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用文件
COPY backend/ ./backend/
COPY frontend/ ./frontend/

WORKDIR /app/backend

# 暴露端口
EXPOSE 5050

# 启动应用
CMD ["python", "app.py"]
