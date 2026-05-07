from openai import OpenAI
import json
import requests


def call_llm(input_message: str, model_name, api_key, base_url):
    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_message},
            ],
            temperature=0.7,
        )
        return response.model_dump()
    except Exception as e:
        return f"Error calling LLM: {str(e)}"


def test_openai_api(api_key, base_url):
    """
    测试 OpenAI 格式 API 的连通性并获取模型列表
    """
    endpoint = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        print(f"正在测试连接: {endpoint}...")
        response = requests.get(endpoint, headers=headers, timeout=10)

        # 检查 HTTP 状态码
        if response.status_code == 200:
            data = response.json()
            # 提取模型 ID 列表
            model_ids = [model["id"] for model in data.get("data", [])]
            print("✅ 连接成功！")
            print(f"可用模型数量: {len(model_ids)}")
            print("前 5 个模型示例:", model_ids[:5])
            return True, model_ids
        else:
            print(f"❌ 连接失败，状态码: {response.status_code}")
            print(f"错误详情: {response.text}")
            return False, response.text

    except requests.exceptions.RequestException as e:
        print(f"⚠️ 网络请求发生异常: {e}")
        return False, str(e)
    
    

if __name__ == "__main__":

    print("Tesing LLM Configs")
    API_CONFIGS = [
        {
            "api_key": "${OPENAI_API_KEY}",
            "base_url": "https://api.siliconflow.cn/v1",
        },
        {
            "api_key": "${IMAGE_MODEL_API_KEY}",
            "base_url": "https://api.gpugeek.com/v1",
        },
        {
            "api_key": "${OPENAI_API_KEY}",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
    ]

    results = []

    for api_config in API_CONFIGS:
        api_key = api_config["api_key"]
        base_url = api_config["base_url"]
        is_works, result = test_openai_api(api_key=api_key, base_url=base_url)
        results.append({"api_key": api_key, "base_url": base_url, "models": result})

    with open("test/test_api_result.json", "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print("Testing LLM Calls")
    result = call_llm(
        input_message="Hello, introduce yourself",
        api_key=API_CONFIGS[0]["api_key"],
        base_url=API_CONFIGS[0]["base_url"],
        model_name="Pro/zai-org/GLM-5",
    )
    print("LLM Response:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
