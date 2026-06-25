import json
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from typing import Optional
import re

CHROMA_DIR  = "data/chroma"
COLLECTION  = "eo_tamil_nadu"
MODEL_NAME  = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "gpt-oss:120b-cloud"
OLLAMA_HOST  = "http://172.19.48.1:11434"  # replace with your actual IP
TOP_K       = 5

# --- Pydantic output schema ---
class EOQueryResult(BaseModel):
    summary: str
    vegetation_status: str
    avg_ndvi: Optional[float]
    hotspot_location: Optional[str]
    confidence: str
    tiles_retrieved: int

# --- Prompt ---
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an Earth Observation analyst. You have been given satellite data
from Sentinel-2 imagery over Tamil Nadu, India.

Use ONLY the context below to answer the question. Be precise and technical.
Return your answer as a JSON object with these exact fields:
- summary: one sentence answer to the question
- vegetation_status: one of [healthy, moderate, sparse, barren, mixed]
- avg_ndvi: average NDVI value from the retrieved tiles (float)
- hotspot_location: lat/lon of most relevant tile (string) or null
- confidence: one of [high, medium, low]
- tiles_retrieved: number of tiles used (int)

Context:
{context}

Question: {question}

Respond with ONLY the JSON object, no explanation, no markdown:"""
)

def load_retriever():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION)
    model = SentenceTransformer(MODEL_NAME)
    return collection, model

def retrieve(query, collection, model):
    embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=embedding,
        n_results=TOP_K,
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    context = "\n".join(docs)
    return context, metas

def parse_output(raw: str) -> EOQueryResult:
    raw = raw.strip()
    # strip markdown fences if model adds them
    raw = re.sub(r"```json|```", "", raw).strip()
    data = json.loads(raw)
    return EOQueryResult(**data)

def query(question: str):
    collection, model = load_retriever()
    llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_HOST)
    chain = PROMPT | llm

    print(f"\nQuery: {question}")
    print("Retrieving tiles...")
    context, metas = retrieve(question, collection, model)

    print("Generating answer...\n")
    raw = chain.invoke({"context": context, "question": question})

    try:
        result = parse_output(raw)
        print("=" * 50)
        print(f"Summary          : {result.summary}")
        print(f"Vegetation status: {result.vegetation_status}")
        print(f"Avg NDVI         : {result.avg_ndvi}")
        print(f"Hotspot location : {result.hotspot_location}")
        print(f"Confidence       : {result.confidence}")
        print(f"Tiles retrieved  : {result.tiles_retrieved}")
        print("=" * 50)
        return result
    except Exception as e:
        print(f"Parse error: {e}")
        print(f"Raw output:\n{raw}")

if __name__ == "__main__":
    questions = [
        "What is the vegetation health status across the Madurai region?",
        "Which areas have the highest NDVI values?",
        "Are there any signs of agricultural activity in the scene?",
    ]
    for q in questions:
        query(q)
        print()
