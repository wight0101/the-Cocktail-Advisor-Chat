import faiss
import numpy as np
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

OPENAI_API_KEY = os.getenv("OPENAI_APIKEY")

client = OpenAI(api_key = OPENAI_API_KEY)
df = pd.read_csv("final_cocktails.csv")
cocktail_names = df['name'].tolist()
cocktail_texts = df['text'].tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(cocktail_texts, convert_to_numpy=True)

dimension = embeddings.shape[1]
cocktail_index = faiss.IndexFlatL2(dimension)
cocktail_index.add(embeddings)

user_memory_index = faiss.IndexFlatL2(dimension)
user_memories = {}

def add_user_preference(user_id, preference_text):
    embedding = model.encode([preference_text], convert_to_numpy=True)
    if user_id not in user_memories:
        user_memories[user_id] = []
    user_memories[user_id].append(preference_text)
    user_memory_index.add(embedding)

def find_similar_cocktails(query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = cocktail_index.search(query_embedding, top_k)
    return [cocktail_names[i] for i in indices[0]]

def find_similar_user_preferences(query, top_k=5):
    if user_memory_index.ntotal == 0:
        return []
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = user_memory_index.search(query_embedding, top_k)
    return [user_memories.get(int(i), "Unknown preference") for i in indices[0]]

def get_user_preferences(user_id):
    return {
        "favorite_cocktails": user_memories.get(user_id, []),
        "favorite_ingredients": []
    }

def get_llm_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def rag_with_user_preferences(user_id, query):
    preferences = get_user_preferences(user_id)
    similar_cocktails = find_similar_cocktails(query)
    similar_user_preferences = find_similar_user_preferences(query)
    
    context = "\n".join(similar_cocktails + similar_user_preferences)
    if preferences["favorite_cocktails"]:
        context += f"\nUser's favorite cocktails: {', '.join(preferences['favorite_cocktails'])}"
    
    prompt = f"Use the following cocktail data as context: {context}\n\n{query}"
    response = get_llm_response(prompt)

    try:
        structured_response = json.loads(response)
        return f"Based on the user's preferences, the best cocktails with {', '.join(preferences['favorite_cocktails'])} would be {', '.join(similar_cocktails)}."
    except json.JSONDecodeError:
        return response
