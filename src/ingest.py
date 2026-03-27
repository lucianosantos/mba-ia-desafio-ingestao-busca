import os
import sys
from pathlib import Path
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()

for k in ("GOOGLE_API_KEY", "GOOGLE_EMBEDDING_MODEL", "DATABASE_URL","PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

current_dir = Path(__file__).parent
PDF_PATH = os.getenv("PDF_PATH")

def validate_pdf_path():
    """Validate if PDF file exists and is readable."""
    if not PDF_PATH:
        print("Error: PDF_PATH environment variable is not set")
        sys.exit(1)
    
    pdf_path = Path(PDF_PATH)
    
    if not pdf_path.exists():
        print(f"Error: PDF file not found at: {pdf_path.absolute()}")
        sys.exit(1)
    
    if not pdf_path.is_file():
        print(f"Error: Path is not a file: {pdf_path.absolute()}")
        sys.exit(1)
    
    if pdf_path.suffix.lower() != '.pdf':
        print(f"Error: File is not a PDF (extension: {pdf_path.suffix})")
        sys.exit(1)
    
    if not os.access(pdf_path, os.R_OK):
        print(f"Error: PDF file is not readable. Check file permissions.")
        sys.exit(1)
    
    print(f"PDF file validated: {pdf_path.absolute()}")
    return pdf_path

def get_documents_from_file():
    """Load and process PDF documents with comprehensive error handling."""
    try:
        print(f"Loading PDF from: {PDF_PATH}")
        loader = PyPDFLoader(str(PDF_PATH))
        docs = loader.load()
        
        if not docs:
            print("Error: PDF loaded but no pages could be extracted")
            print("   This might indicate the PDF is corrupted or encrypted")
            sys.exit(1)
        
        print(f"Successfully loaded {len(docs)} page(s) from PDF")
        
    except FileNotFoundError:
        print(f"Error: PDF file not found at: {PDF_PATH}")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied reading PDF file: {PDF_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading PDF: {str(e)}")
        print(f"   This might be due to PDF corruption, invalid format, or encryption")
        sys.exit(1)
    
    try:
        print("Splitting documents into chunks...")
        splits = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150, 
            add_start_index=False
        ).split_documents(docs)
        
    except Exception as e:
        print(f"Error splitting documents: {str(e)}")
        sys.exit(1)

    if not splits:
        print("Error: No document splits were created")
        print("   The PDF might be empty or contain only images with no extractable text")
        sys.exit(1)
    
    print(f"Created {len(splits)} document chunks")
    
    try:
        enriched = [
            Document(
                page_content=doc.page_content,
                metadata={k: v for k, v in doc.metadata.items() if v not in ("", None)}
            )
            for doc in splits
        ]
    except Exception as e:
        print(f"Error enriching documents: {str(e)}")
        sys.exit(1)

    print("Documents enriched with metadata")
    return enriched

def save_documents(enriched):
    """Save documents to PostgreSQL vector store with error handling."""
    try:
        ids = [f"doc-{i}" for i in range(len(enriched))]
        
        print("Initializing embeddings model...")
        embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL"))
        
        print("Connecting to PostgreSQL vector store...")
        store = PGVector(
            embeddings=embeddings,
            collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
            connection=os.getenv("DATABASE_URL"),
            use_jsonb=True
        )
        
    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        print("   Make sure PostgreSQL is running and credentials are correct")
        sys.exit(1)

    try:
        print(f"Ingesting {len(enriched)} documents into vector store...")
        print("   (This may take a while and consume API tokens)\n")
        
        # This will consume lots of tokens in no time and may hit rate limits
        store.add_documents(documents=enriched, ids=ids)

        # If your api key has low rate limits, uncomment the code below to add documents with delay
        # print(f"Starting ingestion of {len(enriched)} documents with 2s delay between each...")
        # for i, (doc, doc_id) in enumerate(zip(enriched, ids), start=1):
        #     print(f"Ingesting document {i}/{len(enriched)} (ID: {doc_id})...", end="\r")
        #     try:
        #         store.add_documents(documents=[doc], ids=[doc_id])
        #         if i < len(enriched):
        #             time.sleep(2)
        #     except Exception as e:
        #         print(f"\n❌ Error ingesting document {i}: {str(e)}")
        #         sys.exit(1)
        
        print(f"\nIngestion complete! {len(enriched)} documents successfully stored.")
        
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        print("   This might be due to API rate limits or database connection issues")
        sys.exit(1)

def ingest_pdf():
    """Main ingestion workflow with validation and error handling."""
    print("=" * 60)
    print("PDF Ingestao - Semantic Search RAG System")
    print("=" * 60 + "\n")
    
    try:
        validate_pdf_path()
        documents = get_documents_from_file()
        save_documents(documents)
        print("\n" + "=" * 60)
        print("All operations completed successfully!")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\n\nIngestion cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    ingest_pdf()
