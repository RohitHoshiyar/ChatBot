import openai

openai.api_key = "sk-..."  # Replace with OpenAI or Azure key

def generate_gpt_response(prompt: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert fashion assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response['choices'][0]['message']['content']
