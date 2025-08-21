# enhanced_pdf_processor.py
# Fast PDF processing with PyMuPDF and command-line arguments

import os
import glob
import warnings
import chromadb
import argparse
import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import List, Optional, Dict
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Configure environment
warnings.filterwarnings("ignore", message=".*pin_memory.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

print("📄 Enhanced PDF Processing Pipeline with PyMuPDF")

# Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_TOKENS = 1000
CHUNK_OVERLAP = 200
DEFAULT_OUTPUT_DIR = "data"


def parse_arguments():
    """Parse command-line arguments for flexible processing."""
    parser = argparse.ArgumentParser(
        description="Process PDF documents into a searchable vector database using PyMuPDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_pdf_processor.py                              # Process default research_papers folder
  python enhanced_pdf_processor.py -i /path/to/pdfs            # Process specific folder
  python enhanced_pdf_processor.py -i folder1 folder2 folder3  # Process multiple folders
  python enhanced_pdf_processor.py -c my_collection -o custom_db # Custom collection and output
  python enhanced_pdf_processor.py --recursive                 # Include all subfolders
  python enhanced_pdf_processor.py --max-tokens 1500          # Custom chunk size
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        default=["research_papers"],
        help="Input folder(s) containing PDF files (default: research_papers)",
    )

    parser.add_argument(
        "-c",
        "--collection",
        default="documents",
        help="Collection name in vector database (default: documents)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for vector database (default: data)",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subfolders for PDFs",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"Maximum tokens per chunk (default: {MAX_TOKENS})",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Overlap between chunks (default: {CHUNK_OVERLAP})",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing collection if it exists",
    )

    parser.add_argument(
        "--smart-naming",
        action="store_true",
        help="Use smart collection naming based on folder/file names",
    )

    return parser.parse_args()


def find_pdf_files(input_folders: List[str], recursive: bool = False) -> List[str]:
    """Find all PDF files in specified folders."""
    pdf_files = []

    for folder in input_folders:
        folder_path = Path(folder)

        if not folder_path.exists():
            print(f"⚠️  Warning: Folder '{folder}' does not exist, skipping...")
            continue

        if not folder_path.is_dir():
            print(f"⚠️  Warning: '{folder}' is not a directory, skipping...")
            continue

        # Search for PDFs
        if recursive:
            pattern = f"{folder}/**/*.pdf"
            found_files = glob.glob(pattern, recursive=True)
        else:
            pattern = f"{folder}/*.pdf"
            found_files = glob.glob(pattern)

        pdf_files.extend(found_files)
        print(f"📂 Found {len(found_files)} PDFs in {folder}")

    return pdf_files


def extract_headings_from_blocks(blocks: Dict) -> List[str]:
    """Extract potential headings from text blocks based on font size and formatting."""
    headings = []

    for block in blocks.get("blocks", []):
        if block.get("type") == 0:  # Text block
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    font_size = span.get("size", 0)
                    font_flags = span.get("flags", 0)

                    # Consider as heading if:
                    # - Font size > 14 OR
                    # - Bold text (flags & 16) OR
                    # - Text is short and capitalized
                    if (
                        font_size > 14
                        or (font_flags & 16)
                        or (len(text) < 100 and text.isupper())
                    ):
                        if text and len(text.strip()) > 3:
                            headings.append(text)

    return headings[:5]  # Return top 5 headings


def process_pdf_with_pymupdf(pdf_path: str) -> List[Dict]:
    """Extract text and metadata from PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        pages_data = []

        for page_num in range(doc.page_count):
            page = doc[page_num]

            # Extract text
            text = page.get_text()

            # Extract structured data
            blocks = page.get_text("dict")
            headings = extract_headings_from_blocks(blocks)

            # Clean text
            text = re.sub(r"\n+", "\n", text)  # Remove excessive newlines
            text = re.sub(r"\s+", " ", text)  # Normalize whitespace

            pages_data.append(
                {
                    "text": text.strip(),
                    "page_num": page_num + 1,
                    "headings": headings,
                    "char_count": len(text),
                }
            )

        doc.close()
        return pages_data

    except Exception as e:
        print(f"   ❌ Error processing PDF: {str(e)}")
        return []


def chunk_text_content(
    text: str, max_tokens: int = MAX_TOKENS, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Split text into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )

    chunks = splitter.split_text(text)
    return chunks


def process_pdf_to_chunks(
    pdf_path: str, max_tokens: int, chunk_overlap: int
) -> List[Document]:
    """Process a single PDF file into document chunks using PyMuPDF."""
    try:
        filename = os.path.basename(pdf_path)

        # Extract pages data
        pages_data = process_pdf_with_pymupdf(pdf_path)

        if not pages_data:
            return []

        # Combine all text
        full_text = "\n\n".join([page["text"] for page in pages_data if page["text"]])

        # Extract overall headings
        all_headings = []
        for page in pages_data:
            all_headings.extend(page["headings"])

        # Get unique headings
        unique_headings = list(dict.fromkeys(all_headings))[:5]

        # Create chunks
        text_chunks = chunk_text_content(full_text, max_tokens, chunk_overlap)

        documents = []
        for chunk_id, chunk_text in enumerate(text_chunks):
            # Find which pages this chunk likely belongs to
            chunk_pages = []
            chunk_pos = full_text.find(chunk_text)

            if chunk_pos != -1:
                char_count = 0
                for page in pages_data:
                    char_count += len(page["text"])
                    if char_count >= chunk_pos:
                        chunk_pages.append(page["page_num"])
                        break

            # Create document with metadata
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "filename": filename,
                    "filepath": pdf_path,
                    "chunk_id": chunk_id,
                    "total_chunks": len(text_chunks),
                    "page_numbers": str(chunk_pages) if chunk_pages else "[]",
                    "title": unique_headings[0] if unique_headings else "No Title",
                    "all_headings": str(unique_headings),
                    "text_length": len(chunk_text),
                    "source_type": "pdf",
                },
            )
            documents.append(doc)

        return documents

    except Exception as e:
        print(f"   ❌ Error processing {os.path.basename(pdf_path)}: {str(e)}")
        return []


def generate_smart_collection_name(
    pdf_files: List[str], input_folders: List[str]
) -> str:
    """Generate intelligent collection name based on files and folders."""
    if not pdf_files:
        return "documents"

    if len(pdf_files) == 1:
        # Single file: use filename without extension
        return Path(pdf_files[0]).stem.lower().replace(" ", "_")

    # Multiple files: try folder name or common prefix
    if len(input_folders) == 1:
        folder_name = os.path.basename(input_folders[0].rstrip("/\\"))
        if folder_name and folder_name != "." and folder_name != "..":
            return f"{folder_name.lower().replace(' ', '_')}_{len(pdf_files)}"

    # Fallback: use first word of first file + count
    first_filename = Path(pdf_files[0]).stem
    first_word = re.split(r"[_\s-]", first_filename)[0].lower()
    return f"{first_word}_{len(pdf_files)}"


def check_existing_collection(output_dir: str, collection_name: str) -> bool:
    """Check if collection already exists."""
    try:
        db_path = os.path.join(output_dir, "chroma_db")
        if not os.path.exists(db_path):
            return False

        chroma_client = chromadb.PersistentClient(path=db_path)
        collections = chroma_client.list_collections()
        return any(col.name == collection_name for col in collections)
    except:
        return False


def main():
    """Main processing pipeline."""
    args = parse_arguments()

    print(f"🔧 Configuration:")
    print(f"   Input folders: {args.input}")
    print(f"   Collection: {args.collection}")
    print(f"   Output: {args.output}")
    print(f"   Recursive: {args.recursive}")
    print(f"   Max tokens: {args.max_tokens}")
    print(f"   Chunk overlap: {args.chunk_overlap}")
    print(f"   Smart naming: {args.smart_naming}")

    # Find PDF files
    print(f"\n📂 Scanning for PDF files...")
    pdf_files = find_pdf_files(args.input, args.recursive)

    if not pdf_files:
        print("❌ No PDF files found!")
        print("💡 Tips:")
        print("   - Check that the folders exist and contain PDF files")
        print("   - Use --recursive to search subfolders")
        print("   - Specify custom folders with -i /path/to/pdfs")
        return

    print(f"📚 Found {len(pdf_files)} PDF files total")

    # Generate smart collection name if requested
    collection_name = args.collection
    if args.smart_naming:
        collection_name = generate_smart_collection_name(pdf_files, args.input)
        print(f"🎯 Smart collection name: {collection_name}")

    # Check for existing collection
    if check_existing_collection(args.output, collection_name):
        if not args.overwrite:
            print(f"❌ Collection '{collection_name}' already exists!")
            print(
                f"   Use --overwrite to replace it, or choose a different collection name with -c"
            )
            return
        else:
            print(f"⚠️  Overwriting existing collection '{collection_name}'")

    # Initialize models
    print(f"\n🔄 Loading models...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Process PDFs
    print(f"\n🔧 Processing PDFs with PyMuPDF (fast mode)...")
    all_documents = []
    processing_stats = {"successful": 0, "failed": 0, "total_chunks": 0}

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"📄 {i}/{len(pdf_files)}: {os.path.basename(pdf_path)}")

        documents = process_pdf_to_chunks(pdf_path, args.max_tokens, args.chunk_overlap)

        if documents:
            all_documents.extend(documents)
            processing_stats["successful"] += 1
            processing_stats["total_chunks"] += len(documents)
            print(f"   ✅ Created {len(documents)} chunks")
        else:
            processing_stats["failed"] += 1

    # Display processing summary
    print(f"\n📊 PROCESSING SUMMARY:")
    print(f"✅ Successful files: {processing_stats['successful']}/{len(pdf_files)}")
    print(f"❌ Failed files: {processing_stats['failed']}")
    print(f"📄 Total chunks created: {processing_stats['total_chunks']}")

    if processing_stats["successful"] > 0:
        avg_chunks = processing_stats["total_chunks"] / processing_stats["successful"]
        print(f"📊 Average chunks per file: {avg_chunks:.1f}")

    if not all_documents:
        print("❌ No documents processed successfully!")
        return

    # Create vector database
    print(f"\n🔮 Creating vector database...")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    db_path = os.path.join(args.output, "chroma_db")

    # Initialize Chroma client
    chroma_client = chromadb.PersistentClient(path=db_path)

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=embedding_model,
        client=chroma_client,
        collection_name=collection_name,
        persist_directory=db_path,
    )

    print(f"✅ Vector database created successfully!")
    print(f"📊 Total documents in database: {vectorstore._collection.count()}")
    print(f"🗃️ Database location: {db_path}")
    print(f"📝 Collection name: {collection_name}")

    # Performance summary
    print(f"\n⚡ PERFORMANCE SUMMARY:")
    print(f"🚀 Processing method: PyMuPDF (Fast)")
    print(f"📄 Files processed: {processing_stats['successful']}")
    print(f"📝 Total chunks: {processing_stats['total_chunks']}")
    print(f"🔧 Chunk size: {args.max_tokens} tokens")
    print(f"🔗 Chunk overlap: {args.chunk_overlap} tokens")

    print(f"\n🎯 Ready! Use the enhanced_chat_app.py to interact with your documents.")
    print(f"💡 Or run: python enhanced_search_interface.py -c {collection_name}")


if __name__ == "__main__":
    main()
