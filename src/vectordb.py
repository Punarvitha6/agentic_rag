import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from crewai.tools import BaseTool
from config import PDF_PATH, VECTOR_INDEX_DIR, OPENAI_API_KEY, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

class VectorDBManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)

    def create_or_load_index(self):
        if (VECTOR_INDEX_DIR / "index.faiss").exists():
            return FAISS.load_local(str(VECTOR_INDEX_DIR), self.embeddings, allow_dangerous_deserialization=True)
        
        if not PDF_PATH.exists():
            raise FileNotFoundError(f"Missing PDF at {PDF_PATH}. Please add it to src/data/")

        loader = PyPDFLoader(str(PDF_PATH))
        documents = loader.load()

        # Requirement 1.1: Semantic-aware chunking by Headings
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\nIntroduction", "\nGenerative AI options", "\nCustom RAG architectures", "\nRetrievers", "\n\n", "\n", " "],
            keep_separator=True
        )
        
        chunks = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        os.makedirs(VECTOR_INDEX_DIR, exist_ok=True)
        vectorstore.save_local(str(VECTOR_INDEX_DIR))
        return vectorstore

class DocumentSearchTool(BaseTool):
    name: str = "AWS_RAG_Guide_Search"
    description: str = "Search the AWS RAG guide for technical details, page numbers, and architectures."
    
    def _run(self, query: str) -> str:
        manager = VectorDBManager()
        db = manager.create_or_load_index()
        # Requirement 1.2 & 2.0: Top-K Retrieval with Similarity Scores
        docs_with_scores = db.similarity_search_with_relevance_scores(query, k=5)
        
        formatted_results = []
        for doc, score in docs_with_scores:
            page = doc.metadata.get('page', 'Unknown')
            # Log Confidence scores for observability
            content = f"[ID: p{page}][Confidence: {score:.2f}][Page: {page}]\n{doc.page_content}"
            formatted_results.append(content)
            
        return "\n\n---\n\n".join(formatted_results)