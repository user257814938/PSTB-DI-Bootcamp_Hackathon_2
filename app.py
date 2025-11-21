import tempfile
from pathlib import Path
from typing import List
import streamlit as st

from utils import (
    IndexedChunk,
    build_embedder,
    build_faiss_index,
    build_summarizer,
    build_tokenizer,
    chunk_text,
    encode_chunks,
    extract_text,
    search,
    summarize_chunks,
)


def get_cache_dir() -> Path:
    base = Path(tempfile.gettempdir()) / "rag_streamlit"
    base.mkdir(parents=True, exist_ok=True)
    return base


@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return build_tokenizer()


@st.cache_resource(show_spinner=False)
def get_embedder():
    return build_embedder()


@st.cache_resource(show_spinner=False)
def get_summarizer():
    return build_summarizer()


st.set_page_config(page_title="Recherche + Résumé (CPU)", layout="wide")
st.title("🔎 Recherche sémantique + résumé (CPU-friendly)")

st.sidebar.header("📂 Upload & Préparation")
uploaded_file = st.sidebar.file_uploader("Choisir un fichier", type=["txt", "pdf", "docx"])
chunk_size = st.sidebar.slider("Taille des chunks (tokens)", min_value=100, max_value=400, value=250, step=25)
overlap = st.sidebar.slider("Overlap (tokens)", min_value=0, max_value=100, value=30, step=10)
batch_size = st.sidebar.select_slider("Batch embeddings", options=[2, 4, 8], value=4)
top_k = st.sidebar.slider("Top-k résultats", min_value=1, max_value=10, value=5)

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks: List[IndexedChunk] = []
    st.session_state.embeddings = None


def handle_upload():
    if not uploaded_file:
        st.warning("Uploadez un fichier pour démarrer.")
        return

    cache_dir = get_cache_dir()
    dest_path = cache_dir / uploaded_file.name
    with dest_path.open("wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Extraction du texte..."):
        text = extract_text(dest_path)

    tokenizer = get_tokenizer()
    with st.spinner("Découpage en chunks..."):
        chunks_text = chunk_text(text, tokenizer=tokenizer, chunk_size=chunk_size, overlap=overlap)

    if not chunks_text:
        st.error("Aucun texte détecté après découpe.")
        return

    embedder = get_embedder()
    with st.spinner("Calcul des embeddings (CPU)..."):
        embeddings = encode_chunks(chunks_text, embedder=embedder, batch_size=batch_size)

    with st.spinner("Construction de l'index FAISS..."):
        index = build_faiss_index(embeddings)

    st.session_state.index = index
    st.session_state.embeddings = embeddings
    st.session_state.chunks = [IndexedChunk(text=chunk, doc_id=uploaded_file.name, chunk_id=i) for i, chunk in enumerate(chunks_text)]
    st.success(f"Index construit avec {len(chunks_text)} chunks.")


st.sidebar.button("Indexer le document", on_click=handle_upload)

st.subheader("Requête")
query = st.text_input("Texte de la requête")

if st.button("Lancer la recherche"):
    if not st.session_state.index:
        st.error("Aucun index n'est disponible. Uploadez et indexez un document d'abord.")
    elif not query.strip():
        st.warning("La requête est vide.")
    else:
        embedder = get_embedder()
        query_emb = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, idxs = search(query_emb, st.session_state.index, top_k=top_k)
        best_scores = scores[0]
        best_idxs = idxs[0]

        retrieved = []
        for score, idx in zip(best_scores, best_idxs):
            if idx == -1:
                continue
            chunk = st.session_state.chunks[idx]
            retrieved.append((chunk, score))

        if not retrieved:
            st.info("Aucun résultat retourné.")
        else:
            st.markdown("### Résultats")
            for rank, (chunk, score) in enumerate(retrieved, start=1):
                st.write(f"**#{rank}** — distance L2: {score:.4f}")
                st.caption(f"{chunk.doc_id} | chunk {chunk.chunk_id}")
                st.code(chunk.text)

            summarizer = get_summarizer()
            with st.spinner("Génération du résumé (t5-small, CPU)..."):
                summary = summarize_chunks([item[0] for item in retrieved], summarizer=summarizer)
            st.markdown("### Résumé synthétique")
            st.success(summary)
