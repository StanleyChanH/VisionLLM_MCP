#!/usr/bin/env python3
"""
VisionLLM MCP Server 启动脚本
"""

import os
import sys
import logging
from dotenv import load_dotenv

# 添加项目目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """检查必要的环境变量"""
    # 检查是否配置了DashScope API密钥
    dashscope_key = os.getenv('DASHSCOPE_API_KEY')
    
    if not dashscope_key:
        logger.warning("未配置DashScope API密钥")
        logger.info("请设置DASHSCOPE_API_KEY环境变量")
        return False
    
    return True

def main():
    """主函数"""
    logger.info("启动 VisionLLM MCP Server")
    
    # 检查环境变量
    if not check_environment():
        logger.warning("环境变量检查失败，将继续启动（某些功能可能不可用）")
    
    # 导入并运行主服务器
    try:
        from vision_mcp_server import mcp
        port = int(os.getenv("MCP_PORT", 8000))
        host = os.getenv("MCP_HOST", "0.0.0.0")
        
        logger.info(f"服务器将在 {host}:{port} 上运行")
        logger.info("按 Ctrl+C 停止服务器")
        
        mcp.run(host=host, port=port)
    except KeyboardInterrupt:
        logger.info("服务器已停止")
    except Exception as e:
        logger.error(f"启动服务器时出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()