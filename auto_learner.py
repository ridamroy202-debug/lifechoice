import os
import time
from fastapi.testclient import TestClient
from openai import OpenAI

from app.main import app, init_db
from app.settings import get_settings

def run():
    init_db()
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    tc = TestClient(app)
    auth_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzgyMzY2NTY0LCJpYXQiOjE3NzcxODI1NjQsImp0aSI6IjNhNTc0YmMzYzNiMDRjYjBhODYzMGJjYjBmMDE0NDNlIiwidXNlcl9pZCI6IjQifQ.9DtBQ6xFbND1kFJ2kPBtXoHjpCnltCTxW9aBpWWRBB8"

    def answer_prompt(prompt, competency):
        sys_msg = f"You are a skilled professional learning {competency}. Answer the following prompt directly, concisely (1-3 sentences), demonstrating strong practical understanding."
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content

    print("Starting auto-learner...")
    res = tc.post("/session/1158/interact", json={"message": "", "auth_token": auth_token})
    data = res.json()

    while data.get("phase") != "completed":
        prompt = data.get("current_prompt")
        
        comp = data.get("current_competency", "skill")
        print("-------------------------------")
        if not prompt:
            print("No current_prompt. Sending 'Continue' to advance.")
            answer = "Continue"
        else:
            print(f"PROMPT ({data.get('phase')}):\n{prompt}")
            answer = answer_prompt(prompt, comp)
            print(f"\nANSWER:\n{answer}")
        
        res = tc.post("/session/1158/interact", json={"message": answer, "auth_token": auth_token})
        if res.status_code != 200:
            print(f"Error HTTP {res.status_code}: {res.text}")
            break
        data = res.json()
        print(f"\nResult: phase={data.get('phase')}, completed={data.get('gamification', {}).get('competency_progress_percent')}%")
        time.sleep(0.5)

    print("Done! End phase:", data.get("phase"))

if __name__ == "__main__":
    run()
