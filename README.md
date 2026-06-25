# 🛰️ Earth Observation RAG System

Natural language queries over real Sentinel-2 satellite imagery — built on LangChain, ChromaDB, and Ollama.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![LangChain](https://img.shields.io/badge/LangChain-latest-purple) ![ChromaDB](https://img.shields.io/badge/ChromaDB-vector--store-green) ![Sentinel-2](https://img.shields.io/badge/Sentinel--2-Copernicus-blue)

---

## What it does

Ask a question in plain English about land cover and vegetation over Tamil Nadu, India. The system retrieves semantically relevant Sentinel-2 satellite tiles from a vector database and generates a structured, grounded answer — no hallucination, no keyword matching.

**Example queries:**
- *"What is the vegetation health status across the Madurai region?"*
- *"Which areas have the highest NDVI values?"*
- *"Are there any signs of agricultural activity in the scene?"*

**Example output:**
```json
{
  "summary": "Tile T43PHM_0271 near lat 10.26, lon 78.08 has the highest NDVI",
  "vegetation_status": "sparse",
  "avg_ndvi": 0.1345,
  "hotspot_location": "10.261943, 78.088631",
  "confidence": "high",
  "tiles_retrieved": 5
}
```

---

## Architecture

```
Sentinel-2 (CDSE)
      │
      ▼
Band extraction (B04 Red + B08 NIR)
      │
      ▼
NDVI computation per 512×512 tile  →  484 JSON records
      │
      ▼
Text chunking  →  sentence-transformers (MiniLM-L6-v2)  →  ChromaDB
      │
      ▼
Query  →  semantic retrieval (top-5 tiles)  →  LangChain prompt
      │
      ▼
gpt-oss:120b (Ollama)  →  Pydantic validated JSON output
      │
      ▼
Streamlit + Folium UI
```

**Data:** Sentinel-2 L1C, tile T43PHM, 2024-03-28, Madurai/Dindigul region, Tamil Nadu
**Coverage:** ~12,500 km² · 484 tiles · each tile = 5.12 km × 5.12 km (26 km²)

---

## Stack

| Component | Purpose |
|---|---|
| `rasterio` | Read Sentinel-2 .jp2 band files |
| `pyproj` | Reproject UTM → WGS84 coordinates |
| `numpy` | NDVI computation per tile |
| `sentence-transformers` | Embed tile text chunks (MiniLM-L6-v2) |
| `chromadb` | Persistent vector store (cosine similarity) |
| `langchain-core` | Prompt templating and chain composition |
| `langchain-ollama` | Ollama LLM integration |
| `ollama` | Local/cloud LLM runtime |
| `pydantic` | Structured output validation |
| `streamlit` | Web UI |
| `folium` | Interactive map rendering |
| `python-dotenv` | Credentials management |

---

## Project structure

```
eo-rag/
├── .env                      # CDSE credentials (not committed)
├── download_sentinel.py      # Search products zip and Download + extract bands
├── compute_ndvi.py           # Tile NDVI computation → JSON
├── embed_tiles.py            # Text chunking + ChromaDB embedding
├── rag_query.py              # RAG query layer (CLI)
├── app.py                    # Streamlit + Folium UI
├── data/
│   ├── raw/                  # Downloaded zip (gitignored)
│   ├── bands/                # Extracted .jp2 files (gitignored)
│   ├── ndvi/
│   │   └── ndvi_tiles.json   # 484 structured tile records
│   └── chroma/               # ChromaDB vector store
└── requirements.txt
```

---

## Setup

### 1. Register on Copernicus Data Space

Create a free account at [dataspace.copernicus.eu](https://dataspace.copernicus.eu).
Select **User Category: Public** and **Type of user: Research & education organisation**.

### 2. Clone and set up environment

```bash
git clone https://github.com/hpthehacker0/eo-rag.git
cd eo-rag

# WSL / Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Add credentials

```bash
cp .env.example .env
# Edit .env with your CDSE email and password
```

`.env` format:
```
CDSE_USER=your_email@example.com
CDSE_PASSWORD=your_password
```

### 4. Install and configure Ollama

Download Ollama from [ollama.com](https://ollama.com).

```bash
# Windows PowerShell — allow external connections for WSL
$env:OLLAMA_HOST="0.0.0.0"
ollama serve

# Pull the model
ollama pull gpt-oss:120b-cloud
```

> **WSL users:** Find your Windows host IP with `ip route | grep default | awk '{print $3}'` and update `OLLAMA_HOST` in `app.py` and `rag_query.py`.

---

## Running the pipeline

### Phase 1 — Download + NDVI

```bash
# Search available products, Download bands and extract B04/B08
python3 download_sentinel.py

# Compute NDVI tiles → data/ndvi/ndvi_tiles.json
python3 compute_ndvi.py
```

### Phase 2 — Embed into ChromaDB

```bash
python3 embed_tiles.py
```

### Phase 3 — Query via CLI

```bash
python3 rag_query.py
```

### Phase 4 — Launch Streamlit UI

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## How it works

### Why RAG over keyword search

Traditional satellite data tools (QGIS, GEE) require domain knowledge — you specify band numbers, thresholds, and spatial filters manually. This system lets non-specialists query satellite data in plain English.

Keyword search fails because the query "agricultural activity" shares no words with tile text like "bare_soil_or_urban, NDVI 0.089" — even though post-harvest March farmland in Tamil Nadu looks exactly like bare soil. Semantic search via vector embeddings bridges this gap because meaning, not vocabulary, drives retrieval.

### Why not just send everything to the LLM

Sending all 484 tile records as context every query costs ~29,000 tokens, dilutes the LLM's attention, and doesn't scale. RAG retrieves only the 5 most relevant tiles (~300 tokens), keeps the answer grounded in specific evidence, and scales to millions of tiles at the same cost.

### Pydantic output schema

Every answer is validated against a fixed schema at runtime:

```python
class EOQueryResult(BaseModel):
    summary: str
    vegetation_status: str          # healthy | moderate | sparse | barren | mixed
    avg_ndvi: Optional[float]
    hotspot_location: Optional[str] # "lat,lon"
    confidence: str                 # high | medium | low
    tiles_retrieved: int
```

This makes the output machine-readable and pluggable into downstream systems.

---

## Data notes

- **Scene date:** 2024-03-28 (dry season, post-harvest)
- **Cloud cover:** ≤30%
- **Tile size:** 512×512 pixels = 5.12 km × 5.12 km = ~26 km² per tile
- **Total coverage:** ~12,500 km² (Madurai, Dindigul, Theni districts)
- **Distribution:** 457 bare soil/urban · 27 sparse vegetation (expected for March)

---

## Extending the project

- **Add more dates:** Re-run Phase 1 with a different `DATE_START`/`DATE_END` and append to ChromaDB for temporal queries
- **Add more tiles:** Expand `BBOX` coordinates to cover more of Tamil Nadu
- **Add more bands:** Include B11/B12 for SWIR-based drought or moisture indices
- **Add L2A data:** Switch to atmospherically corrected L2A products for more accurate absolute NDVI values


---

## License

MIT

---

*Built as a portfolio project targeting India's space tech industry *
