# Requires PyTorch 2.9.0 or higher as well as https://github.com/vllm-project/vllm/pull/24583
#
# vllm serve Qwen/Qwen3-8B --enforce-eager

import asyncio
import httpx

async def main():
    url = "http://localhost:8000/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        # "model": "Qwen/Qwen3-30B-A3B",
        "model": "Qwen/Qwen3-8B",
        "messages": [
            {
                "role": "user",
                "content": "Generate 1000 random numbers. Go directly into it, don't say Sure and don't say here are numbers. Just start with a number. /no_think",
                
            }
        ],
        "chat_template_kwargs": {
            "thinking": False
        },
        "temperature": 0.0,
        "max_tokens": 100,
    }

    outs = []
    responses = []
    async with httpx.AsyncClient() as client:
        for i in range(1000):
            response = client.post(url, headers=headers, json=data, timeout=120)

            responses.append(response)

        responses = await asyncio.gather(*responses)
        for response in responses:
            outs.append(response.json()['choices'][0]['message']['content'])

    for i in outs:
        print(i.replace("\n", " "))
    print(f"Total samples: {len(outs)}, Unique samples: {len(set(outs))}")

asyncio.run(main())