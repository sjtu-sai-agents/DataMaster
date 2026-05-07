import requests
import json

def test_gpugeek_endpoint(endpoint_path):
    url = f"http://127.0.0.1:9903/v1/{endpoint_path}"
    headers = {
        "Authorization": f"Bearer ${IMAGE_MODEL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "zhipu/GLM-4.5", 
        "messages": [{"role": "user", "content": "Say hello"}]
    }

    print(f"正在测试路径: {url}")
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 404:
            print("❌ 验证结果：该端点不存在 (404 Not Found)。确认 GPUGeek 不支持此路径。")
        elif response.status_code == 200:
            print(response.json())
            print("✅ 验证结果：该端点可用！")
        else:
            print(f"⚠️ 其他返回: {response.text}")
            
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    print("--- 步骤 2: 测试 GPUGeek 标准的 'chat/completions' 路径 ---")
    test_gpugeek_endpoint("chat/completions")
    
    
    print("--- 步骤 1: 测试 Codex CLI 的 'responses' 路径 ---")
    test_gpugeek_endpoint("responses")
    print("\n" + "-"*40 + "\n")
    