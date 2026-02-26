#!/bin/bash
#
# RWKVServe - Test Inference Server
#
# This script sends test requests to a running RWKVServe API server
# to verify it is working correctly.
#
# Prerequisites:
#   The server must already be running (see start_server.sh).
#
# Usage:
#   chmod +x test_server.sh
#   ./test_server.sh
#

set -e

# ==================== Configuration ====================

SERVER_URL="http://localhost:8000"
MODEL_NAME="rwkv-7-grpo"

# ==================== No modification needed below ====================

PASS=0
FAIL=0

print_header() {
    echo ""
    echo "=================================================="
    echo "  $1"
    echo "=================================================="
}

check_result() {
    local test_name="$1"
    local http_code="$2"
    local body="$3"
    local expected_code="${4:-200}"

    if [ "$http_code" = "$expected_code" ]; then
        echo "[PASS] ${test_name} (HTTP ${http_code})"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] ${test_name} (expected HTTP ${expected_code}, got ${http_code})"
        echo "  Response: ${body}"
        FAIL=$((FAIL + 1))
    fi
}

print_header "RWKVServe API Test Suite"
echo ""
echo "Server: ${SERVER_URL}"
echo "Model:  ${MODEL_NAME}"
echo ""

# ----------------------------------------------------------
# Test 1: Health check
# ----------------------------------------------------------
echo "--- Test 1: Health Check ---"
RESPONSE=$(curl -s -w "\n%{http_code}" "${SERVER_URL}/health" 2>/dev/null || echo -e "\n000")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "000" ]; then
    echo "[FAIL] Cannot connect to server at ${SERVER_URL}"
    echo "  Make sure the server is running (see start_server.sh)"
    exit 1
fi
check_result "Health check" "$HTTP_CODE" "$BODY"
echo "  Response: ${BODY}"
echo ""

# ----------------------------------------------------------
# Test 2: List models
# ----------------------------------------------------------
echo "--- Test 2: List Models ---"
RESPONSE=$(curl -s -w "\n%{http_code}" "${SERVER_URL}/v1/models")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

check_result "List models" "$HTTP_CODE" "$BODY"
echo "  Models: ${BODY}" | head -c 500
echo ""
echo ""

# ----------------------------------------------------------
# Test 3: Chat completions (non-streaming)
# ----------------------------------------------------------
echo "--- Test 3: Chat Completions (non-streaming) ---"
RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST "${SERVER_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"${MODEL_NAME}"'",
        "messages": [
            {"role": "user", "content": "Hello! Who are you? Please answer briefly."}
        ],
        "max_tokens": 128,
        "temperature": 0.8,
        "top_p": 0.9
    }')
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

check_result "Chat completions (non-streaming)" "$HTTP_CODE" "$BODY"

CONTENT=$(echo "$BODY" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data['choices'][0]['message']['content'])
except Exception as e:
    print(f'(parse error: {e})')
" 2>/dev/null || echo "(parse error)")
echo "  Assistant: ${CONTENT}"
echo ""

# ----------------------------------------------------------
# Test 4: Chat completions (streaming)
# ----------------------------------------------------------
echo "--- Test 4: Chat Completions (streaming) ---"
STREAM_OUTPUT=$(curl -s -N \
    -X POST "${SERVER_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"${MODEL_NAME}"'",
        "messages": [
            {"role": "user", "content": "Count from 1 to 5."}
        ],
        "max_tokens": 64,
        "temperature": 0.3,
        "stream": true
    }' 2>/dev/null)

CHUNK_COUNT=$(echo "$STREAM_OUTPUT" | grep -c "^data:" || true)
if [ "$CHUNK_COUNT" -gt 0 ]; then
    echo "[PASS] Chat completions (streaming) - received ${CHUNK_COUNT} chunks"
    PASS=$((PASS + 1))

    FULL_TEXT=$(echo "$STREAM_OUTPUT" | grep "^data:" | grep -v "\[DONE\]" | while read -r line; do
        echo "$line" | sed 's/^data: //' | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    delta = data.get('choices', [{}])[0].get('delta', {})
    content = delta.get('content', '')
    if content:
        print(content, end='')
except:
    pass
" 2>/dev/null
    done)
    echo "  Streamed output: ${FULL_TEXT}"
else
    echo "[FAIL] Chat completions (streaming) - no SSE chunks received"
    echo "  Raw output: ${STREAM_OUTPUT}" | head -c 500
    FAIL=$((FAIL + 1))
fi
echo ""

# ----------------------------------------------------------
# Test 5: Text completions
# ----------------------------------------------------------
echo "--- Test 5: Text Completions ---"
RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST "${SERVER_URL}/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"${MODEL_NAME}"'",
        "prompt": "The meaning of life is",
        "max_tokens": 64,
        "temperature": 0.8,
        "top_p": 0.9
    }')
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

check_result "Text completions" "$HTTP_CODE" "$BODY"

COMPLETION=$(echo "$BODY" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data['choices'][0]['text'])
except Exception as e:
    print(f'(parse error: {e})')
" 2>/dev/null || echo "(parse error)")
echo "  Completion: ${COMPLETION}"
echo ""

# ----------------------------------------------------------
# Test 6: Multi-turn conversation
# ----------------------------------------------------------
echo "--- Test 6: Multi-turn Conversation ---"
RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST "${SERVER_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"${MODEL_NAME}"'",
        "messages": [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
            {"role": "user", "content": "What is my name?"}
        ],
        "max_tokens": 64,
        "temperature": 0.3
    }')
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

check_result "Multi-turn conversation" "$HTTP_CODE" "$BODY"

CONTENT=$(echo "$BODY" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data['choices'][0]['message']['content'])
except Exception as e:
    print(f'(parse error: {e})')
" 2>/dev/null || echo "(parse error)")
echo "  Assistant: ${CONTENT}"
echo ""

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
print_header "Test Summary"
echo ""
TOTAL=$((PASS + FAIL))
echo "  Total:  ${TOTAL}"
echo "  Passed: ${PASS}"
echo "  Failed: ${FAIL}"
echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "  All tests passed!"
else
    echo "  Some tests failed. Check the output above for details."
fi
echo ""
echo "=================================================="
