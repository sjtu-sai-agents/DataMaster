import logging
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import requests as req

app = FastAPI(title="Codex CLI → GPUGeek API Conversion Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPUGeek API configuration
GPU_GEEK_BASE_URL = "https://api.gpugeek.com/v1"
GPU_GEEK_API_KEY = "${IMAGE_MODEL_API_KEY}"


def get_gpugeek_headers():
    """Get headers for GPUGeek API requests"""
    return {
        "Authorization": f"Bearer {GPU_GEEK_API_KEY}",
        "Content-Type": "application/json",
    }


def convert_codex_to_gpugeek(codex_request):
    """Convert Codex CLI format to GPUGeek format"""
    # Check if it's already in standard OpenAI format
    if "messages" in codex_request:
        # Already in standard format, just pass through
        gpugeek_request = {
            "model": codex_request.get("model", "zhipu/GLM-4.5"),
            "messages": codex_request.get("messages", []),
            "temperature": codex_request.get("temperature"),
            "max_tokens": codex_request.get("max_tokens"),
            "top_p": codex_request.get("top_p"),
            "stream": codex_request.get("stream", False)
        }
        return {k: v for k, v in gpugeek_request.items() if v is not None}

    # Convert from Codex CLI format
    messages = []

    # Add instructions as system message if present
    if "instructions" in codex_request and codex_request["instructions"]:
        messages.append({
            "role": "system",
            "content": codex_request["instructions"]
        })

    # Process input messages
    if "input" in codex_request and codex_request["input"]:
        for msg in codex_request["input"]:
            if msg.get("type") == "message":
                role = msg.get("role", "user")
                content_list = msg.get("content", [])

                # Extract text content from content list
                text_content = ""
                for content_item in content_list:
                    if content_item.get("type") == "input_text":
                        text_content = content_item.get("text", "")
                        break

                if text_content:
                    messages.append({
                        "role": role,
                        "content": text_content
                    })

    gpugeek_request = {
        "model": codex_request.get("model", "zhipu/GLM-4.5"),
        "messages": messages,
        "temperature": codex_request.get("temperature"),
        "max_tokens": codex_request.get("max_tokens"),
        "top_p": codex_request.get("top_p"),
        "stream": codex_request.get("stream", False)
    }

    # Remove None values
    return {k: v for k, v in gpugeek_request.items() if v is not None}


def convert_gpugeek_to_codex(gpugeek_response):
    """Convert GPUGeek response format to Codex CLI format"""
    if "choices" in gpugeek_response and len(gpugeek_response["choices"]) > 0:
        choice = gpugeek_response["choices"][0]
        return {
            "id": gpugeek_response.get("id", ""),
            "object": "chat.completion",
            "created": gpugeek_response.get("created", 0),
            "model": gpugeek_response.get("model", ""),
            "choices": [
                {
                    "index": choice.get("index", 0),
                    "message": {
                        "role": choice.get("message", {}).get("role", "assistant"),
                        "content": choice.get("message", {}).get("content", ""),
                    },
                    "finish_reason": choice.get("finish_reason", "stop"),
                }
            ],
            "usage": gpugeek_response.get(
                "usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            ),
        }
    return gpugeek_response


@app.post("/v1/responses")
async def handle_responses(request: Request):
    """Handle Codex CLI responses endpoint - convert to GPUGeek chat/completions"""
    try:
        codex_request = await request.json()
        logger.info(f"Received request on /responses: {codex_request}")

        # Convert to GPUGeek format
        gpugeek_request = convert_codex_to_gpugeek(codex_request)
        logger.info(f"Converted to GPUGeek format: {gpugeek_request}")

        # Forward to GPUGeek API
        gpugeek_url = f"{GPU_GEEK_BASE_URL}/chat/completions"

        # Check if streaming is requested
        is_streaming = gpugeek_request.get("stream", False)
        logger.info(f"Streaming mode: {is_streaming}")

        if is_streaming:
            # For streaming, we need to stream the response
            response = req.post(
                gpugeek_url, headers=get_gpugeek_headers(), json=gpugeek_request, timeout=60, stream=True
            )
        else:
            # For non-streaming, regular timeout
            response = req.post(
                gpugeek_url, headers=get_gpugeek_headers(), json=gpugeek_request, timeout=30, stream=False
            )

        logger.info(f"GPUGeek API status: {response.status_code}")

        if response.status_code == 200:
            if is_streaming:
                # Handle streaming response - use SSE format properly
                def generate():
                    try:
                        # Process the SSE stream from upstream
                        for line in response.iter_lines(decode_unicode=True):
                            if line:
                                # Forward the SSE line as-is to maintain format
                                # The upstream already sends "data: {...}" format
                                if line.startswith('data:'):
                                    # Ensure proper SSE format with double newline
                                    yield line + '\n\n'
                                elif line.strip():
                                    # Handle any other non-empty lines
                                    yield line + '\n\n'

                    except GeneratorExit:
                        # Client disconnected
                        logger.info("Stream client disconnected")
                    except Exception as e:
                        logger.error(f"Streaming error: {e}")
                        error_data = {'error': str(e)}
                        yield f"data: {json.dumps(error_data)}\n\n"
                    finally:
                        # Ensure response is properly closed
                        try:
                            response.close()
                        except Exception:
                            pass

                return StreamingResponse(generate(), media_type="text/event-stream")
            else:
                # Handle non-streaming response
                try:
                    gpugeek_response = response.json()
                    codex_response = convert_gpugeek_to_codex(gpugeek_response)
                    logger.info(f"Successfully converted response back to Codex format")
                    return JSONResponse(content=codex_response, status_code=200)
                except Exception as json_error:
                    logger.error(f"JSON parsing error: {json_error}")
                    logger.error(f"Response text: {response.text[:500]}")
                    return JSONResponse(
                        content={"error": f"JSON parsing error: {str(json_error)}", "raw_response": response.text[:1000]},
                        status_code=500,
                    )
        else:
            logger.error(
                f"GPUGeek API returned error: {response.status_code}, {response.text[:500]}"
            )
            return JSONResponse(
                content={"error": f"GPUGeek API error: {response.text[:1000]}"},
                status_code=response.status_code,
            )

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.api_route("/v1/{endpoint_path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_endpoint(endpoint_path: str, request: Request):
    """Proxy all other endpoints to GPUGeek API as-is"""
    try:
        # Construct target URL
        target_url = f"{GPU_GEEK_BASE_URL}/{endpoint_path}"
        logger.info(f"Proxying {request.method} request to: {target_url}")

        # Get request data
        headers = get_gpugeek_headers()
        json_data = (
            await request.json()
            if request.headers.get("content-type", "").startswith("application/json")
            else None
        )
        params = dict(request.query_params)

        # Make the proxy request
        if request.method == "GET":
            response = req.get(target_url, headers=headers, params=params, timeout=30)
        elif request.method == "POST":
            response = req.post(
                target_url, headers=headers, json=json_data, params=params, timeout=30
            )
        elif request.method == "PUT":
            response = req.put(
                target_url, headers=headers, json=json_data, params=params, timeout=30
            )
        elif request.method == "DELETE":
            response = req.delete(
                target_url, headers=headers, params=params, timeout=30
            )
        else:
            return JSONResponse(
                content={"error": "Unsupported method"}, status_code=405
            )

        # Return response
        return JSONResponse(content=response.json(), status_code=response.status_code)

    except Exception as e:
        logger.error(f"Error proxying request: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "codex-conversion-proxy"}


if __name__ == "__main__":
    print("=" * 50)
    print("Codex CLI → GPUGeek API Conversion Service")
    print("=" * 50)
    print(f"GPUGeek Base URL: {GPU_GEEK_BASE_URL}")
    print(f"Conversion Endpoint: /v1/responses → /v1/chat/completions")
    print(f"Proxy Endpoint: All other paths → same path")
    print("=" * 50)

    # Run the server with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9903)
