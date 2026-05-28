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


# Hand-curated common-name aliases for major federal statutes.
# Added at index time so semantic search learns the colloquial name -> section
# mapping for cases where the section_title is generic (e.g. 21 U.S.C. § 841's
# title is just "Prohibited acts A", but it's the main federal drug-trafficking
# statute). Without this, MiniLM ranks more-topically-titled sections higher.
_STATUTE_ALIASES: dict[str, str] = {
    # Title 18 (Crimes)
    "18 U.S.C. § 1030": "computer fraud; CFAA; Computer Fraud and Abuse Act; unauthorized access to protected computer; hacking",
    "18 U.S.C. § 1028": "identity theft; identification fraud; false identification documents",
    "18 U.S.C. § 1028A": "aggravated identity theft; identity theft sentence enhancement",
    "18 U.S.C. § 1029": "access device fraud; credit card fraud; counterfeit access devices",
    "18 U.S.C. § 1037": "CAN-SPAM violations; commercial electronic mail fraud",
    "18 U.S.C. § 1201": "kidnapping; interstate transportation of victim; federal kidnapping statute",
    "18 U.S.C. § 1202": "ransom money; kidnapping ransom",
    "18 U.S.C. § 875": "ransom demand; interstate communication of threats; extortionate threats",
    "18 U.S.C. § 1341": "mail fraud; scheme to defraud via the mails",
    "18 U.S.C. § 1343": "wire fraud; scheme to defraud via interstate wire radio or television communications",
    "18 U.S.C. § 1344": "bank fraud; scheme to defraud a financial institution",
    "18 U.S.C. § 1349": "attempt and conspiracy to commit mail or wire fraud",
    "18 U.S.C. § 1956": "money laundering; concealment of illicit proceeds; financial transaction to conceal source",
    "18 U.S.C. § 1957": "money laundering; monetary transactions in property derived from specified unlawful activity",
    "18 U.S.C. § 2113": "bank robbery; armed robbery of federally insured institution; federal bank robbery statute",
    "18 U.S.C. § 1951": "Hobbs Act; robbery or extortion affecting interstate commerce",
    "18 U.S.C. § 1955": "illegal gambling business",
    "18 U.S.C. § 793": "espionage; gathering or transmitting national defense information",
    "18 U.S.C. § 794": "espionage; communicating defense information to foreign government",
    "18 U.S.C. § 798": "disclosure of classified information; communications intelligence",
    "18 U.S.C. § 924": "firearm offenses; using firearm in crime of violence or drug trafficking; § 924(c)",
    "18 U.S.C. § 1073": "interstate flight to avoid prosecution",
    "18 U.S.C. § 641": "theft of government property",
    "18 U.S.C. § 371": "conspiracy to commit federal offense",
    # Title 21 (Drugs)
    "21 U.S.C. § 841": "drug trafficking; main federal drug crime; manufacturing distributing or dispensing controlled substances; prohibited acts",
    "21 U.S.C. § 846": "drug conspiracy; attempt and conspiracy to violate drug laws",
    "21 U.S.C. § 848": "continuing criminal enterprise; drug kingpin statute",
    "21 U.S.C. § 952": "drug importation; importing controlled substances",
    # Title 26 (Tax)
    "26 U.S.C. § 7201": "tax evasion; attempt to evade or defeat tax",
    "26 U.S.C. § 7202": "willful failure to collect and pay over employment tax; payroll tax fraud",
    "26 U.S.C. § 7203": "willful failure to file tax return",
    "26 U.S.C. § 7206": "false tax return; fraud and false statements on tax filings",
    # Title 31 (Money & Finance)
    "31 U.S.C. § 5313": "currency transaction report; CTR; Bank Secrecy Act reporting",
    "31 U.S.C. § 5324": "structuring transactions; evading currency reporting requirements; BSA structuring",
    # Title 15 (Commerce / Securities)
    "15 U.S.C. § 78j": "securities fraud; manipulative or deceptive devices; Rule 10b-5",
}


def prepare_documents(usc_data: list[dict]) -> list[Document]:
    """
    Convert USC JSON entries to LangChain Document objects.

    Args:
        usc_data: USC data loaded from JSON

    Returns:
        LangChain-compatible documents whose page_content is enriched with
        chapter-context emphasis and (when applicable) a hand-curated alias line.
    """
    print("Preparing documents for vector database...")

    documents = []
    enriched = 0
    for entry in usc_data:
        citation = entry["citation"]
        alias_line = _STATUTE_ALIASES.get(citation, "")
        chapter_title = entry["chapter_title"]

        # Build content. When the section has a curated alias, put it FIRST so
        # the embedding model's early-token attention puts maximum weight on
        # the common-name phrases. The alias is also repeated once for extra
        # signal. Without this, sections whose section_title is generic (e.g.
        # 21 U.S.C. § 841 = "Prohibited acts A") get out-ranked by sections
        # whose title happens to contain topical words.
        lines: list[str] = []
        if alias_line:
            lines.append(alias_line)
            lines.append(f"Common names: {alias_line}")
            lines.append(f"This statute is: {alias_line}")
            enriched += 1
        lines.extend([
            f"Statute: {citation}",
            f"Title {entry['title']} - {entry['title_name']}",
            f"Chapter {entry['chapter']}: {chapter_title}",
            f"Section {entry['Section']}: {entry['section_title']}",
            "",
            entry["section_desc"],
        ])

        doc = Document(
            page_content="\n".join(lines),
            metadata={
                "title": entry["title"],
                "title_name": entry["title_name"],
                "chapter": entry["chapter"],
                "chapter_title": chapter_title,
                "section": entry["Section"],
                "section_title": entry["section_title"],
                "citation": citation,
            },
        )
        documents.append(doc)

    print(f"Prepared {len(documents)} documents ({enriched} with curated aliases)")
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