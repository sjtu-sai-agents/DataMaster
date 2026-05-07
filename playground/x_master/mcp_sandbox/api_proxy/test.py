# end_to_end_test.py
import asyncio
import aiohttp
import json
import sys

# 添加当前目录到路径，以便导入
sys.path.append('.')
from api_utils.web_search_api import serper_google_search

async def test_full_flow():
    print("🔍 端到端测试")
    print("=" * 50)
    
    api_key = "6c89ff344ef65133ee3c197cd33be2bf2e1f5a89"  # 你的有效 key
    
    # 测试1: 直接调用 serper_google_search
    print("\n1. 直接调用 serper_google_search:")
    try:
        result = await serper_google_search(
            query="artificial intelligence",
            serper_api_key=api_key,
            top_k=5,
            region="us",
            lang="en",
            depth=0
        )
        print(f"   ✅ 成功！结果数: {len(result)}")
        if len(result) > 0:
            print(f"   第一个结果标题: {result[0].get('title', '')[:50]}...")
        else:
            print("   ⚠️  返回空数组！")
    except Exception as e:
        print(f"   ❌ 失败: {type(e).__name__}: {e}")
    
    # 测试2: 通过 FastAPI
    print("\n2. 通过 FastAPI 调用:")
    async with aiohttp.ClientSession() as session:
        payload = {
            "query": "artificial intelligence",
            "serper_api_key": api_key,
            "top_k": 5,
            "region": "us",
            "lang": "en",
            "depth": 0
        }
        
        try:
            async with session.post(
                "http://localhost:1234/search",
                json=payload,
                timeout=10
            ) as response:
                print(f"   状态码: {response.status}")
                text = await response.text()
                
                if response.status == 200:
                    try:
                        data = json.loads(text)
                        print(f"   ✅ API 成功！返回类型: {type(data)}")
                        if isinstance(data, list):
                            print(f"   结果数: {len(data)}")
                            if len(data) > 0:
                                print(f"   第一个结果: {data[0].get('title', '')[:50]}...")
                        else:
                            print(f"   返回内容: {text[:200]}")
                    except json.JSONDecodeError:
                        print(f"   ❌ JSON 解析失败，原始响应: {text[:200]}")
                else:
                    print(f"   ❌ HTTP 错误: {text[:200]}")
                    
        except Exception as e:
            print(f"   ❌ 请求失败: {type(e).__name__}: {e}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    asyncio.run(test_full_flow())