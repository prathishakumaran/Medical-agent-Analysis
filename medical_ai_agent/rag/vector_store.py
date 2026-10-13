import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faiss_index")

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_or_update_vector_store(chunks):
    """Creates a new FAISS index from chunks or adds to an existing one."""
    if not chunks:
        return None
        
    embeddings = get_embeddings()
    
    if os.path.exists(DB_DIR):
        try:
            db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(chunks)
            db.save_local(DB_DIR)
            return db
        except Exception as e:
            print(f"Error loading existing index, creating new one: {e}")
            
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_DIR)
    return db

def query_vector_store(query: str, k: int = 3):
    """Searches the FAISS index for relevant medical chunks."""
    if not os.path.exists(DB_DIR):
        return []
        
    embeddings = get_embeddings()
    try:
        db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(query, k=k)
        return docs
    except Exception as e:
        print(f"Error querying index: {e}")
        return []
