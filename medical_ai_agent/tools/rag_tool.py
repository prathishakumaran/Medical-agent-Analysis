from langchain.tools import tool
from rag.vector_store import query_vector_store

@tool
def query_medical_knowledge(query: str) -> str:
    """
    Search the uploaded medical PDF documents for specific medical context, history, or knowledge.
    Use this tool when the user asks a question about a medical report or PDF that has been uploaded to the system.
    """
    docs = query_vector_store(query, k=3)
    if not docs:
        return "No relevant medical documents found in the knowledge base."
    
    response = "Found the following context from medical reports:\n\n"
    for i, doc in enumerate(docs):
        response += f"--- Excerpt {i+1} ---\n{doc.page_content}\n\n"
        
    return response
