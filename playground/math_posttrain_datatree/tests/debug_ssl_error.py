"""Debug script to understand the SSL error with datasets-server.huggingface.co"""

import ssl
import time
import requests
from urllib3.util.ssl_ import create_urllib3_context

print("=" * 80)
print("SSL/TLS Diagnostics")
print("=" * 80)

# 1. Check SSL/TLS versions
print(f"\n1. OpenSSL version: {ssl.OPENSSL_VERSION}")
print(f"   SSL version info: {ssl.OPENSSL_VERSION_INFO}")
print(f"   Requests version: {requests.__version__}")

# 2. Check supported protocols
context = ssl.create_default_context()
print(f"\n2. Default SSL context settings:")
print(f"   Protocol: {context.protocol}")
print(f"   Check hostname: {context.check_hostname}")
print(f"   Verify mode: {context.verify_mode}")

# 3. Test connection to datasets-server.huggingface.co
print(f"\n3. Testing connection to datasets-server.huggingface.co...")
test_url = "https://datasets-server.huggingface.co/splits?dataset=openai/gsm8k"

# Try multiple times to simulate the intermittent error
success_count = 0
fail_count = 0
errors = []

for i in range(10):
    try:
        print(f"\n   Attempt {i+1}/10...", end=" ")
        resp = requests.get(test_url, timeout=10)
        resp.raise_for_status()
        print(f"✓ SUCCESS (status: {resp.status_code})")
        success_count += 1
    except requests.exceptions.SSLError as e:
        print(f"✗ SSL ERROR: {e}")
        fail_count += 1
        errors.append(str(e))
    except Exception as e:
        print(f"✗ OTHER ERROR: {type(e).__name__}: {e}")
        fail_count += 1
        errors.append(str(e))

    time.sleep(0.5)  # Small delay between attempts

print(f"\n4. Results:")
print(f"   Success: {success_count}/10")
print(f"   Failures: {fail_count}/10")

if errors:
    print(f"\n5. Error details:")
    for idx, err in enumerate(errors, 1):
        print(f"   Error {idx}: {err}")

# 6. Check urllib3 settings
print(f"\n6. urllib3 SSL settings:")
try:
    import urllib3
    print(f"   urllib3 version: {urllib3.__version__}")
    print(f"   SSL module: {urllib3.util.ssl_}")
except Exception as e:
    print(f"   Error checking urllib3: {e}")

# 7. Common causes of SSL EOF errors
print(f"\n7. Common causes of 'SSL: UNEXPECTED_EOF_WHILE_READING' error:")
print("""
   a) Server closes connection prematurely during SSL handshake
   b) Network issues (firewall, proxy, unstable connection)
   c) TLS version mismatch between client and server
   d) Load balancer or CDN issues on the server side
   e) Rate limiting causing abrupt connection termination
   f) OpenSSL 3.x stricter validation vs OpenSSL 1.x
   g) Concurrent connections overwhelming the server
""")

# 8. Recommendations
print(f"\n8. Recommended fixes:")
print("""
   Option 1: Add connection pooling with retry adapter
   Option 2: Increase retry attempts and backoff time
   Option 3: Add connection keep-alive headers
   Option 4: Implement exponential backoff with jitter
   Option 5: Use HTTP/1.1 explicitly instead of HTTP/2
   Option 6: Add connection timeout separate from read timeout
""")

print("\n" + "=" * 80)
