import json
import re
import streamlit as st
import folium
from streamlit_folium import st_folium
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

CHROMA_DIR   = "data/chroma"
COLLECTION   = "eo_tamil_nadu"
MODEL_NAME   = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "gpt-oss:120b-cloud"
OLLAMA_HOST  = "http://172.19.48.1:11434"   # your WSL gateway IP
TOP_K        = 5

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an Earth Observation analyst. You have been given satellite data
from Sentinel-2 imagery over Tamil Nadu, India.

Use ONLY the context below to answer the question. Be precise and technical.
Return your answer as a JSON object with these exact fields:
- summary: one sentence answer to the question
- vegetation_status: one of [healthy, moderate, sparse, barren, mixed]
- avg_ndvi: average NDVI value from the retrieved tiles (float)
- hotspot_location: lat/lon of most relevant tile as "lat,lon" string or null
- confidence: one of [high, medium, low]
- tiles_retrieved: number of tiles used (int)

Context:
{context}

Question: {question}

Respond with ONLY the JSON object, no explanation, no markdown:"""
)

NDVI_COLORS = {
    "water_or_cloud"    : "#3b82f6",
    "bare_soil_or_urban": "#d97706",
    "sparse_vegetation" : "#84cc16",
    "moderate_vegetation": "#22c55e",
    "dense_vegetation"  : "#15803d",
}

CONFIDENCE_COLORS = {
    "high"  : "green",
    "medium": "orange",
    "low"   : "red",
}

@st.cache_resource
def load_resources():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION)
    embed_model = SentenceTransformer(MODEL_NAME)
    llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_HOST)
    return collection, embed_model, llm

def retrieve(query, collection, embed_model):
    embedding = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=embedding, n_results=TOP_K)
    return results["documents"][0], results["metadatas"][0]

def parse_output(raw):
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)

def run_rag(question, collection, embed_model, llm):
    docs, metas = retrieve(question, collection, embed_model)
    context = "\n".join(docs)
    chain = PROMPT | llm
    raw = chain.invoke({"context": context, "question": question})
    result = parse_output(raw)
    return result, metas

def build_map(all_tiles, highlight_metas, hotspot):
    m = folium.Map(location=[10.5, 77.9], zoom_start=8, tiles="CartoDB positron")

    # All tiles as small circles
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION)
    all_data = collection.get(include=["metadatas"])
    for meta in all_data["metadatas"]:
        folium.CircleMarker(
            location=[meta["center_lat"], meta["center_lon"]],
            radius=3,
            color=NDVI_COLORS.get(meta["vegetation_class"], "#888"),
            fill=True,
            fill_opacity=0.5,
            tooltip=f"{meta['vegetation_class']} | NDVI: {meta['ndvi_mean']}",
        ).add_to(m)

    # Retrieved tiles highlighted
    for meta in highlight_metas:
        folium.CircleMarker(
            location=[meta["center_lat"], meta["center_lon"]],
            radius=10,
            color="#f59e0b",
            fill=True,
            fill_opacity=0.8,
            tooltip=f"Retrieved: {meta['vegetation_class']} | NDVI: {meta['ndvi_mean']}",
        ).add_to(m)

    # Hotspot marker
    if hotspot:
        try:
            lat, lon = map(float, hotspot.split(","))
            folium.Marker(
                location=[lat, lon],
                tooltip="Hotspot",
                icon=folium.Icon(color="red", icon="star"),
            ).add_to(m)
        except:
            pass

    return m

# --- UI ---
st.set_page_config(page_title="EO RAG — Tamil Nadu", layout="wide")
st.title("🛰️ Earth Observation RAG")
st.caption("Sentinel-2 · Tamil Nadu · March 2024")

col1, col2 = st.columns([1, 1.6])

with col1:
    st.subheader("Query")
    question = st.text_area(
        "Ask about vegetation, land cover, or NDVI:",
        placeholder="e.g. Which areas have the highest vegetation density?",
        height=100,
    )

    examples = [
        "What is the vegetation health across Madurai region?",
        "Which areas have the highest NDVI values?",
        "Are there signs of agricultural activity?",
        "Where is bare soil or urban land cover concentrated?",
    ]
    st.caption("Examples:")
    for ex in examples:
        if st.button(ex, key=ex):
            question = ex

    run = st.button("Run Query", type="primary", disabled=not question)

    if "result" in st.session_state:
        r = st.session_state.result
        st.divider()
        st.subheader("Answer")
        st.markdown(f"**{r['summary']}**")

        c1, c2, c3 = st.columns(3)
        c1.metric("Vegetation", r["vegetation_status"].title())
        c2.metric("Avg NDVI", r.get("avg_ndvi", "N/A"))
        c3.metric("Confidence", r["confidence"].title())

        st.caption(f"Tiles retrieved: {r['tiles_retrieved']}")
        if r.get("hotspot_location"):
            st.caption(f"Hotspot: {r['hotspot_location']}")

with col2:
    st.subheader("Map")

    # Legend
    legend_html = " ".join(
        f'<span style="color:{v}">⬤</span> {k.replace("_"," ")} &nbsp;'
        for k, v in NDVI_COLORS.items()
    )
    st.markdown(legend_html, unsafe_allow_html=True)

    highlight_metas = st.session_state.get("metas", [])
    hotspot = st.session_state.get("result", {}).get("hotspot_location")
    m = build_map([], highlight_metas, hotspot)
    st_folium(m, width=700, height=500)

# Run query
if run and question:
    with st.spinner("Retrieving tiles and generating answer..."):
        collection, embed_model, llm = load_resources()
        result, metas = run_rag(question, collection, embed_model, llm)
        st.session_state.result = result
        st.session_state.metas = metas
    st.rerun()