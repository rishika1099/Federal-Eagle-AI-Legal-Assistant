# usc_vectordb_builder.py
import json
import os
from dotenv import load_dotenv
from langchain_community.docstore.document import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_usc_data(file_path: str) -> list[dict]:
    """
    Load USC data from a JSON file.
    
    Args:
        file_path: Path to the USC JSON file
        
    Returns:
        List of USC sections as dictionaries
    """
    print(f"Loading USC data from: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    print(f"Loaded {len(data)} sections from {len(set(s['title'] for s in data))} titles")
    return data


def prepare_documents(usc_data: list[dict]) -> list[Document]:
    """
    Convert USC JSON entries to LangChain Document objects.
    
    Args:
        usc_data: USC data loaded from JSON
        
    Returns:
        LangChain-compatible documents
    """
    print("Preparing documents for vector database...")
    
    documents = []
    for entry in usc_data:
        # Create rich content for better semantic search
        content = (
            f"Title {entry['title']} - {entry['title_name']}\n"
            f"Chapter {entry['chapter']}: {entry['chapter_title']}\n"
            f"Section {entry['Section']}: {entry['section_title']}\n\n"
            f"{entry['section_desc']}"
        )
        
        doc = Document(
            page_content=content,
            metadata={
                "title": entry["title"],
                "title_name": entry["title_name"],
                "chapter": entry["chapter"],
                "chapter_title": entry["chapter_title"],
                "section": entry["Section"],
                "section_title": entry["section_title"],
                "citation": entry["citation"]
            }
        )
        documents.append(doc)
    
    print(f"Prepared {len(documents)} documents")
    return documents


def build_usc_vectordb():
    """
    Build and persist a Chroma vectorstore for all USC sections.
    """
    load_dotenv()
    
    # Get configuration from environment
    usc_json_path = os.getenv("USC_JSON_PATH", "usc_complete.json")
    persist_dir_path = os.getenv("PERSIST_DIRECTORY_PATH", "./chroma_db")
    collection_name = os.getenv("USC_COLLECTION_NAME", "usc_complete")
    
    print("\nUSC VECTOR DATABASE BUILDER")
    print(f"Input file: {usc_json_path}")
    print(f"Output directory: {persist_dir_path}")
    print(f"Collection name: {collection_name}\n")
    
    # Check if JSON file exists
    if not os.path.exists(usc_json_path):
        raise FileNotFoundError(
            f"USC JSON file not found: {usc_json_path}\n"
            f"Please run 'python usc_parser.py' first to create it"
        )
    
    # Load and process data
    usc_data = load_usc_data(usc_json_path)
    documents = prepare_documents(usc_data)
    
    # Initialize embeddings
    print("Initializing HuggingFace embeddings...")
    print("This may download the model on first run (about 90MB)")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector database
    print("\nBuilding vector database...")
    print("This will take several minutes for large datasets...")
    
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir_path,
        collection_name=collection_name
    )
    
    print(f"\nVector database created successfully!")
    print(f"Collection: {collection_name}")
    print(f"Location: {persist_dir_path}")
    print(f"Total sections: {len(documents)}")
    
    # Test the database
    print("\nTesting vector database with sample queries...")
    test_queries = [
        "computer fraud unauthorized access",
        "murder homicide",
        "tax evasion"
    ]
    
    for query in test_queries:
        print(f"\nTest Query: {query}")
        results = vectordb.similarity_search(query, k=3)
        for i, doc in enumerate(results, 1):
            citation = doc.metadata.get('citation', 'Unknown')
            title = doc.metadata.get('section_title', 'Unknown')
            print(f"  {i}. {citation} - {title}")
    
    print("\nVector database setup complete!")


if __name__ == "__main__":
    build_usc_vectordb()