import os
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()

for k in ("GOOGLE_API_KEY", "GOOGLE_EMBEDDING_MODEL", "PG_VECTOR_URL","PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

current_dir = Path(__file__).parent
PDF_PATH = os.getenv("PDF_PATH")

def ingest_pdf():
    docs = PyPDFLoader(str(PDF_PATH)).load()

    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, add_start_index=False).split_documents(docs)

    if not splits:
        raise RuntimeError("No document splits were created.")

    enriched = [
        Document(
            page_content=doc.page_content,
            metadata={k: v for k, v in doc.metadata.items() if v not in ("", None)}
        )
        for doc in splits
    ]

    ids =[f"doc-{i}" for i in range(len(enriched))]

    embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL"))
    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("PG_VECTOR_URL"),
        use_jsonb=True
    )

    # This will consume lots of tokens in no time and may hit rate limits
    store.add_documents(documents=enriched, ids=ids)

    # If your api key has low rate limits, use the following code to add documents with a delay between each request
    #print(f"Starting ingestion of {len(enriched)} documents with 2s delay between each...")
    #for i, (doc, doc_id) in enumerate(zip(enriched, ids), start=1):
    #    print(f"Ingesting document {i}/{len(enriched)} (ID: {doc_id})...")
    #    store.add_documents(documents=[doc], ids=[doc_id])
    #    if i < len(enriched):  # Don't delay after the last document
    #        print(f"Waiting 2 seconds before next document...")
    #        time.sleep(2)

    print(f"\nIngestion complete!")

if __name__ == "__main__":
    ingest_pdf()
