pip install openai langchain numpy

CHUNK_SIZE = 100
chunks = []

with open("your_file.txt", "r") as f:
    text = f.read()
    current_place = 0
    while current_place < len(text):
        current_chunk = text[current_place:current_place + CHUNK_SIZE]
        chunks.append(current_chunk)
        current_place += CHUNK_SIZE


import openai
openai.api_key = "API KEY"


embedded_chunks = []
for chunk in chunks:
    response = openai.Embedding.create(input=chunk, model="text-embedding-ada-002")
    embedded_chunks.append(response['data'][0]['embedding'])

import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


question = "Why was the NBA replay center created?"
question_embedding = openai.Embedding.create(input=question, model="text-embedding-ada-002")['data'][0]['embedding']

best_chunk = None
best_similarity = -1
for i, chunk_embedding in enumerate(embedded_chunks):
    similarity = cosine_similarity(question_embedding, chunk_embedding)
    if similarity > best_similarity:
        best_similarity = similarity
        best_chunk = chunks[i]
print("Relevant information:", best_chunk)


completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Here is some information: {best_chunk}. Now, answer the question: {question}"}
    ]
)
print(completion['choices'][0]['message']['content'])