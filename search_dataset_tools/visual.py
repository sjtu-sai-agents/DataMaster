"""
多模态图片数据检查 MCP 工具

提供两个主要功能:
1. inspect_images_details: 使用 AI 模型分析图片内容
2. inspect_images_info: 获取图片的物理信息（尺寸、亮度、对比度等）
"""

import base64
import json
import os
from pathlib import Path
from typing import List, Union
import numpy as np
from PIL import Image
from openai import OpenAI
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("visual_tools")



def encode_image(image_path: str) -> str:
    """将图片文件编码为 base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_image_files(image_path: str) -> List[str]:
    """
    获取图片文件列表

    Args:
        image_path: 文件路径或文件夹路径

    Returns:
        图片文件的绝对路径列表
    """
    path = Path(image_path)

    if path.is_file():
        # 单个文件
        if path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            return [str(path.absolute())]
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Only .png, .jpg, .jpeg are supported.")

    elif path.is_dir():
        # 文件夹，获取所有图片
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            image_files.extend(path.glob(ext))

        if not image_files:
            raise ValueError(f"No image files found in directory: {image_path}")

        return [str(f.absolute()) for f in image_files]

    else:
        raise ValueError(f"Path does not exist: {image_path}")


def calculate_brightness(image: Image.Image) -> float:
    """计算图片的平均亮度"""
    # 转换为灰度图
    if image.mode != 'L':
        gray_image = image.convert('L')
    else:
        gray_image = image

    # 计算平均亮度
    np_image = np.array(gray_image)
    return float(np.mean(np_image))


def calculate_contrast(image: Image.Image) -> float:
    """计算图片的对比度（标准差）"""
    # 转换为灰度图
    if image.mode != 'L':
        gray_image = image.convert('L')
    else:
        gray_image = image

    # 计算标准差作为对比度指标
    np_image = np.array(gray_image)
    return float(np.std(np_image))


def get_file_size(image_path: str) -> int:
    """获取文件大小（字节）"""
    return os.path.getsize(image_path)


@mcp.tool()
def inspect_images_details(image_files: Union[str, List[str]], query: str) -> str:
    """
    此工具使用 OpenAI 的多模态模型来理解和分析图片内容，支持单张或多张图片的批量分析。

    **适用场景:**
    - 图片内容描述和识别
    - 图片中的文字提取 (OCR)
    - 图片质量评估
    - 数据可视化图表解读
    - 图片中的数据标注验证

    Args:
        image_files: 图片文件路径，可以是单个文件路径（str）或文件路径列表（List[str]）
        query: 用户的问题或分析要求，例如："描述这张图片的内容"、"提取图片中的所有文字"等

    Returns:
        JSON 字符串，包含 AI 模型的分析结果

    Example:
        >>> inspect_images_details("photo.jpg", "描述这张图片的内容")
        >>> inspect_images_details(["img1.jpg", "img2.png"], "比较这两张图片的差异")
    """
    model_name = "Vendor2/GPT-5.4"
    api_key = os.environ.get("IMAGE_MODEL_API_KEY")
    base_url = os.environ.get("IMAGE_MODEL_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        # 规范化输入为列表
        if isinstance(image_files, str):
            image_paths = get_image_files(image_files)
        else:
            # 处理列表输入
            image_paths = []
            for img_file in image_files:
                image_paths.extend(get_image_files(img_file))

        if not image_paths:
            return json.dumps({
                "error": "No valid image files found",
                "status": "failed"
            }, ensure_ascii=False, indent=2)

        # 构建消息内容
        content = [{"type": "input_text", "text": query}]

        # 添加图片
        for img_path in image_paths:
            base64_image = encode_image(img_path)
            # 根据 MIME 类型
            ext = Path(img_path).suffix.lower()
            mime_type = "image/png" if ext == '.png' else "image/jpeg"

            content.append({
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{base64_image}",
            })

        # 调用 OpenAI API
        system_prompt = """
你是一个专业的图片分析助手，具有以下能力：
1. 准确描述图片中的内容和场景
2. 识别并提取图片中的文字信息
3. 分析图片的质量、构图和视觉效果
4. 解读数据可视化图表（如柱状图、折线图、散点图等）
5. 验证数据标注的准确性
6. 比较多张图片之间的差异和相似性
请用清晰、准确、专业的语言回答用户的问题。如果涉及数据或图表，请尽可能提取具体数值。"""

        response = client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": content
                }
            ]
        )

        result = {
            "status": "success",
            "query": query,
            "images_analyzed": len(image_paths),
            "image_paths": image_paths,
            "analysis": response.output_text
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "status": "failed"
        }, ensure_ascii=False, indent=2)


@mcp.tool()
def inspect_images_info(image_path: str) -> str:
    """
    [图片物理信息提取] 批量获取图片的物理属性信息

    提取图片的尺寸、亮度、对比度、文件大小等物理属性，并支持统计分析。
    适用于数据集质量检查、图片筛选、批量处理前的预分析等场景。

    **功能特性:**
    - 支持单文件或整个文件夹的批量处理
    - 提取尺寸、宽高比、通道数、亮度、对比度、文件大小
    - 批量模式下提供统计信息（平均值、中位数、方差等）
    - 支持 PNG、JPG、JPEG 格式

    Args:
        image_path: 图片文件路径或包含图片的文件夹路径

    Returns:
        JSON 字符串，包含每张图片的详细信息和批量统计信息

    Example:
        >>> inspect_images_info("dataset/train/")
        >>> inspect_images_info("sample.jpg")
    """
    try:
        image_paths = get_image_files(image_path)

        if not image_paths:
            return json.dumps({
                "error": "No valid image files found",
                "status": "failed"
            }, ensure_ascii=False, indent=2)

        results = []

        # 收集每张图片的信息
        for img_path in image_paths:
            try:
                with Image.open(img_path) as img:
                    info = {
                        "path": img_path,
                        "filename": Path(img_path).name,
                        "format": img.format,
                        "mode": img.mode,  # RGB, L, etc.
                        "size": {
                            "width": img.width,
                            "height": img.height,
                            "aspect_ratio": round(img.width / img.height, 4)
                        },
                        "channels": len(img.getbands()),
                        "brightness": round(calculate_brightness(img), 2),
                        "contrast": round(calculate_contrast(img), 2),
                        "file_size_bytes": get_file_size(img_path),
                        "file_size_mb": round(get_file_size(img_path) / (1024 * 1024), 4)
                    }
                    results.append(info)
            except Exception as e:
                results.append({
                    "path": img_path,
                    "error": f"Failed to process: {str(e)}"
                })

        # 如果只有一张图片，直接返回
        if len(results) == 1:
            return json.dumps({
                "status": "success",
                "image": results[0]
            }, ensure_ascii=False, indent=2)

        # 批量统计
        valid_results = [r for r in results if "error" not in r]

        if valid_results:
            # 提取数值用于统计
            widths = [r["size"]["width"] for r in valid_results]
            heights = [r["size"]["height"] for r in valid_results]
            aspect_ratios = [r["size"]["aspect_ratio"] for r in valid_results]
            brightness_values = [r["brightness"] for r in valid_results]
            contrast_values = [r["contrast"] for r in valid_results]
            file_sizes = [r["file_size_bytes"] for r in valid_results]

            statistics = {
                "total_images": len(results),
                "successfully_processed": len(valid_results),
                "failed": len(results) - len(valid_results),
                "statistics": {
                    "width": {
                        "min": min(widths),
                        "max": max(widths),
                        "mean": round(np.mean(widths), 2),
                        "median": round(np.median(widths), 2),
                        "std": round(np.std(widths), 2)
                    },
                    "height": {
                        "min": min(heights),
                        "max": max(heights),
                        "mean": round(np.mean(heights), 2),
                        "median": round(np.median(heights), 2),
                        "std": round(np.std(heights), 2)
                    },
                    "aspect_ratio": {
                        "min": round(min(aspect_ratios), 4),
                        "max": round(max(aspect_ratios), 4),
                        "mean": round(np.mean(aspect_ratios), 4),
                        "median": round(np.median(aspect_ratios), 4),
                        "std": round(np.std(aspect_ratios), 4)
                    },
                    "brightness": {
                        "min": round(min(brightness_values), 2),
                        "max": round(max(brightness_values), 2),
                        "mean": round(np.mean(brightness_values), 2),
                        "median": round(np.median(brightness_values), 2),
                        "std": round(np.std(brightness_values), 2)
                    },
                    "contrast": {
                        "min": round(min(contrast_values), 2),
                        "max": round(max(contrast_values), 2),
                        "mean": round(np.mean(contrast_values), 2),
                        "median": round(np.median(contrast_values), 2),
                        "std": round(np.std(contrast_values), 2)
                    },
                    "file_size_bytes": {
                        "min": min(file_sizes),
                        "max": max(file_sizes),
                        "mean": round(np.mean(file_sizes), 2),
                        "median": round(np.median(file_sizes), 2),
                        "std": round(np.std(file_sizes), 2)
                    }
                }
            }
        else:
            statistics = {
                "total_images": len(results),
                "successfully_processed": 0,
                "failed": len(results),
                "error": "All images failed to process"
            }

        return json.dumps({
            "status": "success",
            "summary": statistics,
            "images": results
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "status": "failed"
        }, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()
