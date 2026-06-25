import json
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

NDVI_JSON = "data/ndvi/ndvi_tiles.json"
CHROMA_DIR = "data/chroma"
COLLECTION = "eo_tamil_nadu"
MODEL_NAME  = "all-MiniLM-L6-v2"  # fast, 384-dim, good for structured text

def tile_to_text(tile):
    """Convert a tile's stats into a natural language chunk for embedding."""
    return (
        f"Satellite tile {tile['tile_id']} observed on {tile['date']}. "
        f"Location: latitude {tile['center_lat']}, longitude {tile['center_lon']}. "
        f"Vegetation analysis: mean NDVI is {tile['ndvi_mean']} "
        f"(min: {tile['ndvi_min']}, max: {tile['ndvi_max']}, std: {tile['ndvi_std']}). "
        f"Land cover classification: {tile['vegetation_class'].replace('_', ' ')}. "
        f"Valid pixel count: {tile['valid_pixels']}."
    )

def main():
    print("Loading NDVI tiles...")
    with open(NDVI_JSON) as f:
        tiles = json.load(f)
    print(f"Loaded {len(tiles)} tiles")

    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded")

    print("\nInitializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Collection '{COLLECTION}' ready")

    print("\nChunking tiles into text...")
    documents = [tile_to_text(t) for t in tiles]
    ids       = [t["tile_id"] for t in tiles]
    metadatas = [
        {
            "date"             : t["date"],
            "center_lat"       : t["center_lat"],
            "center_lon"       : t["center_lon"],
            "ndvi_mean"        : t["ndvi_mean"],
            "vegetation_class" : t["vegetation_class"],
        }
        for t in tiles
    ]

    print(f"Sample chunk:\n  {documents[0]}\n")

    print("Embedding and storing in ChromaDB (batch of 100)...")
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs  = documents[i:i+batch_size]
        batch_ids   = ids[i:i+batch_size]
        batch_meta  = metadatas[i:i+batch_size]
        batch_embeds = model.encode(batch_docs).tolist()

        collection.add(
            documents=batch_docs,
            embeddings=batch_embeds,
            metadatas=batch_meta,
            ids=batch_ids,
        )
        print(f"  Stored batch {i//batch_size + 1} ({min(i+batch_size, len(documents))}/{len(documents)} tiles)")

    print(f"\nDone. Total vectors in collection: {collection.count()}")

    print("\nRunning test query...")
    query = "areas with sparse vegetation near Madurai"
    query_embed = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embed,
        n_results=3,
    )
    print(f"Query: '{query}'")
    print("Top 3 results:")
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\n  [{i+1}] {meta['vegetation_class']} | NDVI: {meta['ndvi_mean']} | lat: {meta['center_lat']}, lon: {meta['center_lon']}")
        print(f"       {doc[:120]}...")

if __name__ == "__main__":
    main()
