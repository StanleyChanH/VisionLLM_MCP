import asyncio
import base64
from fastmcp import FastMCP
import os
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import dashscope
from dashscope import MultiModalConversation

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化FastMCP服务器
mcp = FastMCP("VisionLLM_MCP")

# 初始化DashScope客户端
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")

if not dashscope_api_key:
    logger.warning("未设置DASHSCOPE_API_KEY环境变量")
    raise ValueError("必须设置DASHSCOPE_API_KEY环境变量")

dashscope.api_key = dashscope_api_key

def encode_image(image_path: str) -> Optional[str]:
    """
    将图像文件编码为base64字符串
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        base64编码的图像字符串，如果出错则返回None
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(image_path):
            logger.error(f"图像文件不存在: {image_path}")
            return None
            
        # 检查文件大小（DashScope限制为20MB）
        file_size = os.path.getsize(image_path)
        if file_size > 20 * 1024 * 1024:  # 20MB
            logger.error(f"图像文件过大: {file_size} bytes")
            return None
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"编码图像时出错: {str(e)}")
        return None

def is_url(path: str) -> bool:
    """
    判断路径是否为URL
    
    Args:
        path: 路径字符串
        
    Returns:
        如果是URL返回True，否则返回False
    """
    return path.startswith("http://") or path.startswith("https://")

@mcp.tool(
    name="analyze_image",
    description="分析图像内容，提供详细的视觉描述",
    parameters={
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "图像文件的路径或URL"
            },
            "query": {
                "type": "string",
                "description": "关于图像的查询问题",
                "default": "请描述这张图片的内容"
            },
            "model": {
                "type": "string",
                "enum": ["openai", "dashscope"],
                "description": "使用的模型提供商",
                "default": "dashscope"
            },
            "detail": {
                "type": "string",
                "enum": ["low", "high"],
                "description": "图像分析的详细程度（仅适用于OpenAI）",
                "default": "low"
            }
        },
        "required": ["image_path"]
    }
)
def analyze_image(
    image_path: str, 
    query: str = "请描述这张图片的内容"
) -> Dict[str, Any]:
    """
    使用视觉模型分析图像
    
    Args:
        image_path: 图像文件的路径或URL
        query: 关于图像的查询问题
        model: 使用的模型提供商 ("openai" 或 "dashscope")
        detail: 图像分析的详细程度 ("low" 或 "high")，仅适用于OpenAI
        
    Returns:
        包含分析结果的字典
    """
    try:
        logger.info(f"开始分析图像: {image_path}")
        
        # 准备图像输入
        if is_url(image_path):
            image_input = image_path
        else:
            # 对于本地文件，需要先编码为base64
            base64_image = encode_image(image_path)
            if not base64_image:
                return {
                    "success": False,
                    "error": f"无法读取或编码图像文件: {image_path}"
                }
            image_input = f"data:image/jpeg;base64,{base64_image}"
    
        # 调用DashScope的QwenVL模型
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": query},
                    {"image": image_input}
                ]
            }
        ]
        
        response = MultiModalConversation.call(
            model='qwen-vl-plus',
            messages=messages
        )
        
        if response.status_code == 200:
            result = response.output.choices[0]['message']['content']
            logger.info(f"DashScope图像分析完成: {image_path}")
            
            return {
                "success": True,
                "result": result,
                "image_path": image_path,
                "model": "qwen-vl-plus"
            }
        else:
            return {
                "success": False,
                "error": f"DashScope API调用失败: {response.message}",
                "status_code": response.status_code
            }
    except Exception as e:
        error_msg = f"图像分析过程中出现错误: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg
        }

def analyze_image_with_openai(image_path: str, query: str, detail: str) -> Dict[str, Any]:
    """
    使用OpenAI的GPT-4 Vision模型分析图像
    
    Args:
        image_path: 图像文件的路径
        query: 关于图像的查询问题
        detail: 图像分析的详细程度 ("low" 或 "high")
        
    Returns:
        包含分析结果的字典
    """
    if not openai_client:
        return {
            "success": False,
            "error": "未配置OpenAI API密钥"
        }
        
    # 编码图像
    base64_image = encode_image(image_path)
    if not base64_image:
        return {
            "success": False,
            "error": f"无法读取或编码图像文件: {image_path}"
        }
    
    # 调用OpenAI的视觉模型
    response = openai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": detail
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    
    result = response.choices[0].message.content
    logger.info(f"OpenAI图像分析完成: {image_path}")
    
    return {
        "success": True,
        "result": result,
        "image_path": image_path,
        "model": "gpt-4-vision-preview"
    }

def analyze_image_with_dashscope(image_path: str, query: str) -> Dict[str, Any]:
    """
    使用DashScope的QwenVL模型分析图像
    
    Args:
        image_path: 图像文件的路径或URL
        query: 关于图像的查询问题
        
    Returns:
        包含分析结果的字典
    """
    if not dashscope_api_key:
        return {
            "success": False,
            "error": "未配置DashScope API密钥"
        }
        
    # 准备图像输入
    if is_url(image_path):
        image_input = image_path
    else:
        # 对于本地文件，需要先编码为base64
        base64_image = encode_image(image_path)
        if not base64_image:
            return {
                "success": False,
                "error": f"无法读取或编码图像文件: {image_path}"
            }
        image_input = f"data:image/jpeg;base64,{base64_image}"
    
    # 调用DashScope的QwenVL模型
    messages = [
        {
            "role": "user",
            "content": [
                {"text": query},
                {"image": image_input}
            ]
        }
    ]
    
    response = MultiModalConversation.call(
        model='qwen-vl-plus',
        messages=messages
    )
    
    if response.status_code == 200:
        result = response.output.choices[0]['message']['content']
        logger.info(f"DashScope图像分析完成: {image_path}")
        
        return {
            "success": True,
            "result": result,
            "image_path": image_path,
            "model": "qwen-vl-plus"
        }
    else:
        return {
            "success": False,
            "error": f"DashScope API调用失败: {response.message}",
            "status_code": response.status_code
        }

@mcp.tool(
    name="analyze_image_from_context",
    description="基于上下文中的图像进行分析",
    parameters={
        "type": "object",
        "properties": {
            "context": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string"},
                        "content": {"type": "string"}
                    }
                },
                "description": "对话上下文"
            },
            "image_path": {
                "type": "string",
                "description": "图像文件的路径或URL"
            },
            "query": {
                "type": "string",
                "description": "关于图像的查询问题"
            },
            "model": {
                "type": "string",
                "enum": ["openai", "dashscope"],
                "description": "使用的模型提供商",
                "default": "dashscope"
            },
            "detail": {
                "type": "string",
                "enum": ["low", "high"],
                "description": "图像分析的详细程度（仅适用于OpenAI）",
                "default": "low"
            }
        },
        "required": ["context", "image_path", "query"]
    }
)
def analyze_image_from_context(
    context: List[Dict[str, Any]], 
    image_path: str, 
    query: str
) -> Dict[str, Any]:
    """
    结合上下文信息分析图像内容
    
    Args:
        context: 对话上下文
        image_path: 图像文件的路径或URL
        query: 关于图像的查询问题
        model: 使用的模型提供商 ("openai" 或 "dashscope")
        detail: 图像分析的详细程度 ("low" 或 "high")，仅适用于OpenAI
        
    Returns:
        包含分析结果的字典
    """
    try:
        logger.info(f"开始结合上下文分析图像: {image_path}")
        
        # 准备图像输入
        if is_url(image_path):
            image_input = image_path
        else:
            # 对于本地文件，需要先编码为base64
            base64_image = encode_image(image_path)
            if not base64_image:
                return {
                    "success": False,
                    "error": f"无法读取或编码图像文件: {image_path}"
                }
            image_input = f"data:image/jpeg;base64,{base64_image}"
    
        # 构建上下文消息
        context_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
        
        # 调用DashScope的QwenVL模型
        messages = [
            {
                "role": "system",
                "content": "你是一个视觉分析助手，能够分析图像并结合上下文提供详细的信息。请用中文回答。"
            },
            {
                "role": "user",
                "content": [
                    {"text": f"基于以下对话上下文分析图像：\n{context_text}\n\n请回答：{query}"},
                    {"image": image_input}
                ]
            }
        ]
        
        response = MultiModalConversation.call(
            model='qwen-vl-plus',
            messages=messages
        )
        
        if response.status_code == 200:
            result = response.output.choices[0]['message']['content']
            logger.info(f"结合上下文的DashScope图像分析完成: {image_path}")
            
            return {
                "success": True,
                "result": result,
                "image_path": image_path,
                "model": "qwen-vl-plus"
            }
        else:
            return {
                "success": False,
                "error": f"DashScope API调用失败: {response.message}",
                "status_code": response.status_code
            }
    except Exception as e:
        error_msg = f"图像分析过程中出现错误: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg
        }

def analyze_image_from_context_with_openai(
    context: List[Dict[str, Any]], 
    image_path: str, 
    query: str, 
    detail: str
) -> Dict[str, Any]:
    """
    使用OpenAI结合上下文信息分析图像内容
    
    Args:
        context: 对话上下文
        image_path: 图像文件的路径
        query: 关于图像的查询问题
        detail: 图像分析的详细程度 ("low" 或 "high")
        
    Returns:
        包含分析结果的字典
    """
    if not openai_client:
        return {
            "success": False,
            "error": "未配置OpenAI API密钥"
        }
        
    # 编码图像
    base64_image = encode_image(image_path)
    if not base64_image:
        return {
            "success": False,
            "error": f"无法读取或编码图像文件: {image_path}"
        }
    
    # 构建上下文字符串
    context_messages = []
    for msg in context:
        context_messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })
    
    # 添加系统消息
    messages = [{
        "role": "system",
        "content": "你是一个视觉分析助手，能够分析图像并结合上下文提供详细的信息。请用中文回答。"
    }]
    
    # 添加上下文消息
    messages.extend(context_messages)
    
    # 添加当前查询和图像
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": query},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": detail
                }
            }
        ]
    })
    
    # 调用OpenAI的视觉模型，结合上下文
    response = openai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=1000
    )
    
    result = response.choices[0].message.content
    logger.info(f"结合上下文的OpenAI图像分析完成: {image_path}")
    
    return {
        "success": True,
        "result": result,
        "image_path": image_path,
        "model": "gpt-4-vision-preview"
    }

def analyze_image_from_context_with_dashscope(
    context: List[Dict[str, Any]], 
    image_path: str, 
    query: str
) -> Dict[str, Any]:
    """
    使用DashScope结合上下文信息分析图像内容
    
    Args:
        context: 对话上下文
        image_path: 图像文件的路径或URL
        query: 关于图像的查询问题
        
    Returns:
        包含分析结果的字典
    """
    if not dashscope_api_key:
        return {
            "success": False,
            "error": "未配置DashScope API密钥"
        }
        
    # 准备图像输入
    if is_url(image_path):
        image_input = image_path
    else:
        # 对于本地文件，需要先编码为base64
        base64_image = encode_image(image_path)
        if not base64_image:
            return {
                "success": False,
                "error": f"无法读取或编码图像文件: {image_path}"
            }
        image_input = f"data:image/jpeg;base64,{base64_image}"
    
    # 构建上下文消息
    context_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
    
    # 调用DashScope的QwenVL模型
    messages = [
        {
            "role": "system",
            "content": "你是一个视觉分析助手，能够分析图像并结合上下文提供详细的信息。请用中文回答。"
        },
        {
            "role": "user",
            "content": [
                {"text": f"基于以下对话上下文分析图像：\n{context_text}\n\n请回答：{query}"},
                {"image": image_input}
            ]
        }
    ]
    
    response = MultiModalConversation.call(
        model='qwen-vl-plus',
        messages=messages
    )
    
    if response.status_code == 200:
        result = response.output.choices[0]['message']['content']
        logger.info(f"结合上下文的DashScope图像分析完成: {image_path}")
        
        return {
            "success": True,
            "result": result,
            "image_path": image_path,
            "model": "qwen-vl-plus"
        }
    else:
        return {
            "success": False,
            "error": f"DashScope API调用失败: {response.message}",
            "status_code": response.status_code
        }

@mcp.tool(
    name="list_supported_image_formats",
    description="列出支持的图像格式",
    parameters={
        "type": "object",
        "properties": {}
    }
)
def list_supported_image_formats() -> Dict[str, Any]:
    """
    列出支持的图像格式
    
    Returns:
        包含支持格式列表的字典
    """
    return {
        "success": True,
        "formats": ["jpeg", "jpg", "png", "webp", "gif"],
        "max_size_mb": 20
    }

@mcp.tool(
    name="check_image_file",
    description="检查图像文件是否存在和可访问",
    parameters={
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "图像文件的路径"
            }
        },
        "required": ["image_path"]
    }
)
def check_image_file(image_path: str) -> Dict[str, Any]:
    """
    检查图像文件是否存在和可访问
    
    Args:
        image_path: 图像文件的路径
        
    Returns:
        包含检查结果的字典
    """
    try:
        # 如果是URL，直接返回成功
        if is_url(image_path):
            return {
                "success": True,
                "image_path": image_path,
                "type": "url"
            }
            
        path = Path(image_path)
        if not path.exists():
            return {
                "success": False,
                "error": f"文件不存在: {image_path}"
            }
            
        if not path.is_file():
            return {
                "success": False,
                "error": f"路径不是文件: {image_path}"
            }
            
        # 检查文件扩展名
        suffix = path.suffix.lower()
        supported_formats = [".jpeg", ".jpg", ".png", ".webp", ".gif"]
        if suffix not in supported_formats:
            return {
                "success": False,
                "error": f"不支持的图像格式: {suffix}。支持的格式: {supported_formats}"
            }
            
        # 检查文件大小
        size = path.stat().st_size
        max_size = 20 * 1024 * 1024  # 20MB
        if size > max_size:
            return {
                "success": False,
                "error": f"文件过大: {size} bytes，最大支持: {max_size} bytes"
            }
            
        return {
            "success": True,
            "image_path": image_path,
            "size": size,
            "format": suffix
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"检查文件时出错: {str(e)}"
        }

if __name__ == "__main__":
    # 设置服务器端口
    port = int(os.getenv("MCP_PORT", 8000))
    host = os.getenv("MCP_HOST", "0.0.0.0")
    
    logger.info(f"启动VisionLLM MCP服务器在 {host}:{port}")
    # 运行MCP服务器
    mcp.run(host=host, port=port)