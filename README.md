# VisionLLM MCP Server

一个基于Model Context Protocol (MCP)的视觉增强服务器，为文本大模型提供视觉能力。

## 功能介绍

该MCP服务器实现了以下功能：

1. 接收文本大模型传递的上下文信息
2. 调用阿里云DashScope的QwenVL模型分析图像
3. 将分析结果返回给文本大模型，增强其视觉理解能力

## 安装依赖

推荐使用 `uv` 进行依赖管理：

```bash
# 安装 uv (如果尚未安装)
pip install uv

# 安装项目依赖
uv pip install -e .
```

或者使用传统的 pip：

```bash
pip install -e .
```

## 环境变量配置

需要设置以下环境变量：

```bash
# 复制示例配置文件
cp .env.example .env

# 编辑 .env 文件，填入你的API密钥
```

必须的环境变量：
- `DASHSCOPE_API_KEY`: 你的阿里云DashScope API密钥

可选的环境变量：
- `MCP_HOST`: MCP服务器监听的主机地址（默认: 0.0.0.0）
- `MCP_PORT`: MCP服务器监听的端口（默认: 8000）

## 启动服务器

有几种方式可以启动服务器：

### 直接运行

```bash
python vision_mcp_server.py
```

### 使用启动脚本

```bash
python start_server.py
```

### 使用Docker

```bash
# 构建镜像
docker build -t visionllm-mcp .

# 运行容器
docker run -p 8000:8000 -v ./images:/app/images --env-file .env visionllm-mcp
```

### 使用Docker Compose

```bash
# 启动服务
docker-compose up

# 后台启动服务
docker-compose up -d
```

## 使用方法

服务器提供多个工具供文本大模型调用：

### 1. analyze_image

直接分析图像内容

参数：
- `image_path` (必需): 图像文件的路径或URL
- `query` (可选): 关于图像的查询问题，默认为"请描述这张图片的内容"

返回：
```json
{
  "success": true,
  "result": "图像分析结果",
  "image_path": "/path/to/image.jpg",
  "model": "qwen-vl-plus"
}
```

### 2. analyze_image_from_context

结合上下文分析图像内容

参数：
- `context` (必需): 对话上下文数组
- `image_path` (必需): 图像文件的路径或URL
- `query` (必需): 关于图像的查询问题

返回：
```json
{
  "success": true,
  "result": "结合上下文的图像分析结果",
  "image_path": "/path/to/image.jpg",
  "model": "qwen-vl-plus"
}
```

### 3. list_supported_image_formats

列出支持的图像格式

返回：
```json
{
  "success": true,
  "formats": ["jpeg", "jpg", "png", "webp", "gif"],
  "max_size_mb": 20
}
```

### 4. check_image_file

检查图像文件是否存在和可访问

参数：
- `image_path` (必需): 图像文件的路径

返回：
```json
{
  "success": true,
  "image_path": "/path/to/image.jpg",
  "size": 102400,
  "format": ".jpg"
}
```

## 工作原理

1. 文本大模型在识别到需要视觉理解的意图后，将最近的上下文传递给MCP服务器
2. MCP服务器接收上下文和图像路径
3. 服务器使用DashScope的QwenVL模型进行图像分析
4. 分析结果返回给文本大模型，增强其视觉能力

## 示例

文本大模型可以这样调用MCP工具：

### 示例1：直接分析图像

```json
{
  "tool": "analyze_image",
  "arguments": {
    "image_path": "/app/images/sample.jpg",
    "query": "请描述这张图片中的场景和物体"
  }
}
```

### 示例2：结合上下文分析图像

```json
{
  "tool": "analyze_image_from_context",
  "arguments": {
    "context": [
      {"role": "user", "content": "我刚刚在公园里拍了一张照片"},
      {"role": "assistant", "content": "好的，请分享照片让我看看"}
    ],
    "image_path": "/app/images/park.jpg",
    "query": "根据我们的对话，你能告诉我这张图片中有什么吗？"
  }
}
```

## 支持的模型

### DashScope QwenVL模型

- `qwen-vl-plus`: 平衡效果和成本的视觉语言模型
- 支持本地文件路径和URL两种图像输入方式
- 对中文场景优化更好

## 支持的图像格式

- JPEG / JPG
- PNG
- WEBP
- GIF

文件大小限制：20MB

对于DashScope模型，还支持直接使用图像URL。

## 错误处理

所有工具都会返回结构化的响应，包含`success`字段指示操作是否成功。如果操作失败，会包含`error`字段描述错误原因。

## 配置说明

### Docker部署

使用Docker部署时，可以将本地的图像目录挂载到容器中：

```yaml
volumes:
  - ./images:/app/images
```

这样容器就可以访问主机上的图像文件了。

### 网络配置

默认情况下，服务器监听所有网络接口（0.0.0.0:8000），可以通过环境变量修改：

```bash
MCP_HOST=127.0.0.1
MCP_PORT=8080
```

## 日志

服务器会输出运行日志到标准输出，包含操作信息和错误信息，便于调试和监控。