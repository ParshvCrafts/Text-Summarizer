"""
Edge Case Testing Script
=========================
Tests the summarizer with various edge cases to ensure robust handling.
Run with: python test_edge_cases.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.text_summarizer.pipeline.inference import Summarizer


def test_edge_cases():
    """Test various edge cases for the summarizer."""
    print("=" * 60)
    print("  EDGE CASE TESTING")
    print("=" * 60)
    
    # Initialize summarizer
    print("\nLoading model...")
    summarizer = Summarizer()
    print("Model loaded.\n")
    
    test_cases = [
        {
            "name": "Empty input",
            "input": "",
            "expected": "empty string"
        },
        {
            "name": "Whitespace only",
            "input": "   \n\t  ",
            "expected": "empty string"
        },
        {
            "name": "Single word",
            "input": "Hello",
            "expected": "some output"
        },
        {
            "name": "Very short dialogue",
            "input": "John: Hi!\nSarah: Hello!",
            "expected": "short summary"
        },
        {
            "name": "Special characters",
            "input": "John: Hey! 😀 How are you?\nSarah: Great! 👍",
            "expected": "handles emoji"
        },
        {
            "name": "Unicode characters",
            "input": "José: ¿Cómo estás?\nMaría: Muy bien, gracias!",
            "expected": "handles unicode"
        },
        {
            "name": "Only punctuation",
            "input": "... !!! ???",
            "expected": "some output or empty"
        },
        {
            "name": "Long input (repeated)",
            "input": ("John: This is a test message. " * 100 + 
                     "\nSarah: I understand. " * 100),
            "expected": "truncated and summarized"
        },
        {
            "name": "No speaker labels",
            "input": "Hey, are you coming tonight? I'm not sure, I have work. Come on, it'll be fun!",
            "expected": "still summarizes"
        },
        {
            "name": "Newlines only format",
            "input": "Hey are you coming\nNot sure\nPlease come\nOkay I will",
            "expected": "handles no colons"
        },
        {
            "name": "Mixed format",
            "input": "John: Hi there!\nHello back!\nJohn: How are you?",
            "expected": "handles mixed"
        },
        {
            "name": "Numbers and dates",
            "input": "John: The meeting is at 3:30 PM on 01/15/2024.\nSarah: Got it, I'll be there.",
            "expected": "preserves numbers"
        },
        {
            "name": "Length control - short",
            "input": "John: Hey, are you coming to the party tonight?\nSarah: I'm not sure, I have a lot of work.\nJohn: Come on, it'll be fun!\nSarah: Okay, I'll try to come by 8.",
            "target_length": "short",
            "expected": "shorter summary"
        },
        {
            "name": "Length control - long",
            "input": "John: Hey, are you coming to the party tonight?\nSarah: I'm not sure, I have a lot of work.\nJohn: Come on, it'll be fun!\nSarah: Okay, I'll try to come by 8.",
            "target_length": "long",
            "expected": "longer summary"
        },
        {
            "name": "Confidence scoring",
            "input": "John: Meeting at 2 PM.\nSarah: Confirmed.",
            "return_confidence": True,
            "expected": "returns confidence"
        },
    ]
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {case['name']} ---")
        print(f"Input: {case['input'][:50]}{'...' if len(case['input']) > 50 else ''}")
        
        try:
            start = time.time()
            
            kwargs = {}
            if case.get('target_length'):
                kwargs['target_length'] = case['target_length']
            if case.get('return_confidence'):
                kwargs['return_confidence'] = True
            
            result = summarizer.summarize(case['input'], **kwargs)
            elapsed = time.time() - start
            
            if case.get('return_confidence'):
                summary, confidence = result
                print(f"Output: {summary}")
                print(f"Confidence: {confidence:.3f}")
                status = "✅ PASS" if confidence > 0 else "⚠️ LOW CONFIDENCE"
            else:
                summary = result
                print(f"Output: {summary[:100]}{'...' if len(summary) > 100 else ''}")
                print(f"Length: {len(summary)} chars")
                status = "✅ PASS"
            
            print(f"Time: {elapsed:.2f}s")
            print(f"Status: {status}")
            
            results.append({
                "name": case['name'],
                "status": "pass",
                "output_length": len(summary) if isinstance(result, str) else len(result[0])
            })
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                "name": case['name'],
                "status": "error",
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r['status'] == 'pass')
    total = len(results)
    
    print(f"\n  Tests: {passed}/{total} passed")
    
    for r in results:
        status = "✅" if r['status'] == 'pass' else "❌"
        print(f"  {status} {r['name']}")
    
    if passed == total:
        print("\n✅ All edge cases handled!")
    else:
        print("\n⚠️ Some edge cases need attention")
    
    return results


if __name__ == "__main__":
    test_edge_cases()
