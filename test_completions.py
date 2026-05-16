import sys
from core.llm_client import LLMClient

def test():
    client = LLMClient(agent_name="llama-70b")
    print(f"Testing model: {client.model_name}")
    try:
        response = client._call_api(
            use_completions=True,
            model=client.model_name,
            prompt="You are a helpful assistant.\n\nWhat is 2+2?\n\nAnswer:",
            temperature=0.0,
            max_tokens=50
        )
        print("Success! Output:")
        print(response.choices[0].text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test()
