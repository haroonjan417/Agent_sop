import os
import glob
import tempfile
from crewai.tools import tool
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_DIR = "./chroma_sop_db"
SEED_DIR = "./data/sops"
vectorstore = None
_embeddings = None


def get_embeddings():
    """Local HuggingFace BGE embeddings (lightweight CPU model), loaded once."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _embeddings


def _reset_store():
    """Drops the existing collection so re-indexing doesn't create duplicate chunks."""
    global vectorstore
    try:
        store = vectorstore
        if store is None and os.path.exists(DB_DIR):
            store = Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())
        if store is not None:
            store.delete_collection()
    except Exception:
        pass
    vectorstore = None


def _index_documents(documents) -> None:
    """Chunks documents and (re)builds the ChromaDB index."""
    global vectorstore
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    _reset_store()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=DB_DIR,
    )


def build_vectorstore_from_files(uploaded_files) -> int:
    """Chunks uploaded PDF/TXT files and indexes them into ChromaDB."""
    documents = []

    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif file_extension == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
            else:
                continue

            docs = loader.load()
            for doc in docs:
                doc.metadata["source_name"] = uploaded_file.name
            documents.extend(docs)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not documents:
        return 0

    _index_documents(documents)
    return len(uploaded_files)


def _index_seed_sops() -> bool:
    """Indexes the bundled SOPs in data/sops so the agent works before any upload."""
    documents = []
    for path in glob.glob(os.path.join(SEED_DIR, "*.txt")):
        docs = TextLoader(path, encoding="utf-8").load()
        for doc in docs:
            doc.metadata["source_name"] = os.path.basename(path)
        documents.extend(docs)
    if not documents:
        return False
    _index_documents(documents)
    return True


def query_sop_vectorstore(query: str, k: int = 3) -> str:
    """Queries ChromaDB and returns relevant chunks with source document names."""
    global vectorstore

    if vectorstore is None:
        if os.path.exists(DB_DIR):
            vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())
        elif not _index_seed_sops():
            return "No SOP documents indexed in the vector store yet."

    results = vectorstore.similarity_search(query, k=k)

    if not results:
        return "No relevant SOP procedures found matching this incident."

    formatted_context = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source_name", "Uploaded_SOP_Document")
        formatted_context.append(
            f"--- SOP Chunk {i} [Document Source: {source}] ---\n{doc.page_content}"
        )

    return "\n\n".join(formatted_context)


@tool("SOP Policy Search Tool")
def sop_search_rag(query: str) -> str:
    """Searches the company SOP knowledge base (RAG) and returns the most relevant
    procedure excerpts, each labelled with its source document name. Input should be
    a short natural-language description of the incident or the policy topic."""
    return query_sop_vectorstore(query)
