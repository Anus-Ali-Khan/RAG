import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path="docs"):
    """Loads all text files from the docs directory."""
    print(f"Loading documents from {docs_path}...")

    #Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory {docs_path} does not exist. Please create it and add your company files.")

    #Load all text files from txt directory
    loader = DirectoryLoader(
        path=docs_path, 
        glob="**/*.txt",
        loader_cls=TextLoader)
    

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
    return documents



def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Splits documents into smaller chunks with overlap."""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
        )
    
    chunks = text_splitter.split_documents(documents)

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Creates and persist ChromaDB vector store"""
    print("Creating embeddings and storig it in ChromaDB...")

    # embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
    


    #Create ChromaDB vector store
    print("--- Creating vector store ---")
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space" : "cosine"}
    )

    print("--- Finished creating vector store ---")

    print(f"Vector store created and persist to {persist_directory}...")

    return vector_store




def main():
    print ("Main Function")

    #1. Loading the files
    documents = load_documents(docs_path="docs")

    #2. Chunking the files
    chunks = split_documents(documents)

    #3. Embedding and Storing in vector DB
    vector_store = create_vector_store(chunks)

if __name__ == "__main__":
    main()