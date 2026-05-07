import requests
import json

def test_streaming():
    url = "http://127.0.0.1:9903/v1/responses"
    headers = {
        "Content-Type": "application/json"
    }

    # Test streaming request
    payload = {
        "model": "zhipu/GLM-4.6",
        "messages": [{"role": "user", "content": "Say hello"}],
        "stream": True  # Enable streaming
    }

    print("Testing streaming response...")
    print(f"URL: {url}")
    print(f"Request: {payload}")
    print("-" * 50)

    try:
        response = requests.post(url, headers=headers, json=payload, stream=True)
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")
        print("-" * 50)

        if response.status_code == 200:
            print("Streaming response:")
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    print(line_str)
        else:
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"Exception: {e}")

def test_non_streaming():
    url = "http://127.0.0.1:9903/v1/responses"
    headers = {
        "Content-Type": "application/json"
    }

    # Test non-streaming request
    payload = {
        "model": "zhipu/GLM-4.6",
        "messages": [{"role": "user", "content": "Say hello"}],
        "stream": False  # Disable streaming
    }

    print("\nTesting non-streaming response...")
    print(f"URL: {url}")
    print(f"Request: {payload}")
    print("-" * 50)

    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"Status Code: {response.status_code}")
        print("-" * 50)

        if response.status_code == 200:
            print("Non-streaming response:")
            result = response.json()
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_streaming()
    test_non_streaming()