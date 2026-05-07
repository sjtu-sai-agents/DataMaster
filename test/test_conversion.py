import requests as req
import sys
import os

# Add the test directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'test'))

# Import the conversion function
from codex_conversion import convert_codex_to_gpugeek, convert_gpugeek_to_codex, get_gpugeek_headers

GPUGEEK_BASE_URL = "https://api.gpugeek.com/v1"

def test_conversion():
    """Test the complete conversion process"""

    # Simulate a Codex CLI request
    codex_request = {
        'model': 'zhipu/GLM-4.6',
        'instructions': 'You are a helpful assistant.',
        'input': [
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'Hello'
                    }
                ]
            }
        ]
    }

    print("Original Codex request:")
    print(codex_request)
    print()

    # Convert to GPUGeek format
    try:
        gpugeek_request = convert_codex_to_gpugeek(codex_request)
        print("Converted GPUGeek request:")
        print(gpugeek_request)
        print()
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test the API call
    try:
        print("Testing GPUGeek API...")
        response = req.post(
            f"{GPUGEEK_BASE_URL}/chat/completions",
            headers=get_gpugeek_headers(),
            json=gpugeek_request,
            timeout=30
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            gpugeek_response = response.json()
            print("GPUGeek response received successfully")

            # Convert back to Codex format
            codex_response = convert_gpugeek_to_codex(gpugeek_response)
            print("Final Codex response:")
            print(codex_response)
        else:
            print(f"API Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Error during API call: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_conversion()