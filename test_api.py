"""
API Test Script
===============
Tests all FastAPI endpoints for the Text Summarizer API.
Run with: python test_api.py

Note: Start the API server first with: python app.py
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"


def test_root():
    """Test root endpoint."""
    print("Testing GET /")
    try:
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        print(f"  ✅ PASS: {data}")
        return True
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def test_health():
    """Test health endpoint."""
    print("Testing GET /health")
    try:
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print(f"  ✅ PASS: {data}")
        return True
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def test_summarize():
    """Test single summarization endpoint."""
    print("Testing POST /summarize")
    try:
        payload = {
            "text": "John: Hey, are you coming to the party tonight?\nSarah: I'm not sure, I have work.\nJohn: Come on, it'll be fun!\nSarah: Okay, I'll try to come by 8.",
            "max_length": 128,
            "num_beams": 4
        }
        response = requests.post(f"{BASE_URL}/summarize", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert len(data["summary"]) > 0
        print(f"  ✅ PASS: Summary = '{data['summary'][:50]}...'")
        return True
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def test_summarize_batch():
    """Test batch summarization endpoint."""
    print("Testing POST /summarize/batch")
    try:
        payload = {
            "texts": [
                "Alice: Hi Bob!\nBob: Hello Alice!",
                "Tom: What time is the meeting?\nJane: 3 PM."
            ],
            "max_length": 64,
            "num_beams": 2
        }
        response = requests.post(f"{BASE_URL}/summarize/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "summaries" in data
        assert data["count"] == 2
        print(f"  ✅ PASS: Generated {data['count']} summaries")
        return True
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def test_summarize_empty():
    """Test error handling for empty input."""
    print("Testing POST /summarize with empty text")
    try:
        payload = {"text": ""}
        response = requests.post(f"{BASE_URL}/summarize", json=payload)
        # Should return 422 validation error
        assert response.status_code == 422
        print(f"  ✅ PASS: Correctly rejected empty input (422)")
        return True
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def test_docs():
    """Test API documentation endpoint."""
    print("Testing GET /docs")
    try:
        response = requests.get(f"{BASE_URL}/docs")
        assert response.status_code == 200
        print(f"  ✅ PASS: API docs accessible")
        return True
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def main():
    print("=" * 50)
    print("  TEXT SUMMARIZER - API TESTS")
    print("=" * 50)
    print(f"\nBase URL: {BASE_URL}")
    print("Make sure the API server is running!\n")
    
    # Check if server is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to API server!")
        print("   Start the server with: python app.py")
        return 1
    
    tests = [
        ("Root Endpoint", test_root),
        ("Health Check", test_health),
        ("Single Summarization", test_summarize),
        ("Batch Summarization", test_summarize_batch),
        ("Empty Input Validation", test_summarize_empty),
        ("API Documentation", test_docs),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        results.append(test_func())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 50)
    print(f"  RESULTS: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("\n✅ All API tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
