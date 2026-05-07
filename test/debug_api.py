import requests as req

GPUGEEK_BASE_URL = "https://api.gpugeek.com/v1"
GPU_GEEK_API_KEY = "${IMAGE_MODEL_API_KEY}"

def test_gpugeek_api():
    """Test GPUGeek API directly"""
    headers = {
        "Authorization": f"Bearer {GPU_GEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    # Test with a simple request
    test_request = {
        "model": "zhipu/GLM-4.6",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }

    print("Testing GPUGeek API...")
    print(f"Request: {test_request}")

    try:
        response = req.post(
            f"{GPUGEEK_BASE_URL}/chat/completions",
            headers=headers,
            json=test_request,
            timeout=30
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

        if response.status_code == 200:
            data = response.json()
            print(f"Success! Response data: {data}")
        else:
            print(f"Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Exception occurred: {e}")

if __name__ == "__main__":
    test_gpugeek_api()