from llm_utils import query_llm, LlmApi

def example_bot(messages: list):
    system_prompt = "Respond in a friendly limerick."
    new_messages = query_llm(messages=messages, system_prompt=system_prompt)
    return dict(new_messages=new_messages)

if __name__ == "__main__":
    LlmApi.singleton().init_api(api_key=None, base_url=None)
    messages = [
        {"role": "user", "content": "What's 1 + 1?"},
    ]
    output = example_bot(messages)
    print("== OUTPUT ==")
    print(output)
