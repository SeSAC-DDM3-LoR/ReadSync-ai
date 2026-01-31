import requests
import json
import os

BASE_URL = "http://localhost:8001/api/v1"

def test_classify():
    print("--- Testing Classify ---")
    
    # CASE 1: Independent content question
    req1 = {
        "user_msg": "Who is the main character?",
        "chapter_id": 1
    }
    res1 = requests.post(f"{BASE_URL}/chat/classify", json=req1)
    print(f"Req: {req1['user_msg']}")
    print(f"Res: {res1.json()}")
    
    # CASE 2: Context dependent question
    req2 = {
        "user_msg": "Why did he do that?",
        "chapter_id": 1
    }
    res2 = requests.post(f"{BASE_URL}/chat/classify", json=req2)
    print(f"Req: {req2['user_msg']}")
    print(f"Res: {res2.json()}")
    
    print("\n")

def test_rewrite():
    print("--- Testing Rewrite Query ---")
    
    history = [
        {"role": "user", "content": "Tell me about Harry Potter."},
        {"role": "assistant", "content": "Harry Potter is a wizard who survived Lord Voldemort's attack."}
    ]
    
    req = {
        "user_msg": "Who are his best friends?",
        "previous_messages": history
    }
    
    res = requests.post(f"{BASE_URL}/chat/rewrite-query", json=req)
    print(f"History: {history[-1]['content']}")
    print(f"User Msg: {req['user_msg']}")
    print(f"Rewritten: {res.json()}")

if __name__ == "__main__":
    try:
        test_classify()
        test_rewrite()
    except Exception as e:
        print(f"Test failed: {e}")
