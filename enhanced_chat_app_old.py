# enhanced_chat_app.py

# Conditional SQLite fix for Streamlit Cloud (Linux only)
import sys
import platform

if platform.system() == "Linux":
    try:
        import pysqlite3

        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        pass  # pysqlite3 not available, use system sqlite3

import streamlit as st
import chromadb
import os
import warnings
import tempfile
import hashlib
import pandas as pd
import time
import zipfile
import fitz
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Configure environment
warnings.filterwarnings("ignore", message=".*pin_memory.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_TOKENS = 1000
CHUNK_OVERLAP = 200

# --------------------------------------------------------------
# PDF PROCESSING FUNCTIONS WITH PYMUPDF
# --------------------------------------------------------------


def extract_headings_from_blocks(blocks: List[Dict]) -> List[str]:
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

    return headings[:3]  # Return top 3 headings


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
        st.error(f"Error processing PDF: {str(e)}")
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


def process_single_pdf(file_path: str, filename: str) -> List[Document]:
    """Process a single PDF file into document chunks."""
    try:
        # Extract pages data
        pages_data = process_pdf_with_pymupdf(file_path)

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
        text_chunks = chunk_text_content(full_text)

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

            # Create document
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "filename": filename,
                    "filepath": file_path,
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
        st.error(f"Error processing {filename}: {str(e)}")
        return []


# --------------------------------------------------------------
# FILE HANDLING FUNCTIONS
# --------------------------------------------------------------


def extract_pdfs_from_folder(uploaded_files) -> List[Tuple[str, str, bytes]]:
    """Extract PDF files from uploaded folder structure."""
    pdf_files = []

    for uploaded_file in uploaded_files:
        file_path = uploaded_file.name

        # Check if it's a PDF file
        if file_path.lower().endswith(".pdf"):
            pdf_files.append(
                (file_path, os.path.basename(file_path), uploaded_file.getvalue())
            )

    return pdf_files


def generate_smart_collection_name(
    filenames: List[str], upload_type: str = "files"
) -> str:
    """Generate intelligent collection name based on files."""
    if not filenames:
        return f"collection_{int(time.time())}"

    if len(filenames) == 1:
        # Single file: use filename without extension
        return Path(filenames[0]).stem.lower().replace(" ", "_")

    # Multiple files: use first word of first file + count
    try:
        first_filename = Path(filenames[0]).stem
        first_word = re.split(r"[_\s-]", first_filename)[0].lower()
        # Clean the first word to make it safe for collection name
        first_word = re.sub(r"[^a-zA-Z0-9]", "", first_word)
        if first_word and len(first_word) > 1:
            return f"{first_word}_{len(filenames)}"
        else:
            return f"documents_{len(filenames)}"
    except (IndexError, AttributeError):
        return f"collection_{len(filenames)}"


@st.cache_resource
def load_models():
    """Initialize language model and embedding model."""
    llm = ChatGroq(
        # model="llama3-8b-8192", temperature=0.1, api_key=os.getenv("GROQ_API_KEY")
        model="llama-3.1-8b-instant",
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return llm, embedding_model


def process_uploaded_files(
    uploaded_files: List, embedding_model, upload_type: str = "files"
) -> List[Document]:
    """Process uploaded PDF files into document chunks."""
    all_documents = []

    # Filter only PDF files
    pdf_files = [
        (f.name, f.name, f.getvalue())
        for f in uploaded_files
        if f.name.lower().endswith(".pdf")
    ]

    if not pdf_files:
        st.error("No PDF files found in the upload!")
        return []

    # Create detailed progress tracking
    main_progress = st.progress(0)
    status_container = st.container()

    total_files = len(pdf_files)

    with status_container:
        overall_status = st.empty()
        current_file_status = st.empty()
        stats_display = st.empty()

    total_chunks_created = 0
    successful_files = 0

    for i, (file_path, filename, file_content) in enumerate(pdf_files):
        try:
            # Update progress
            overall_percentage = (i / total_files) * 100
            overall_status.markdown(
                f"**📊 Processing: {overall_percentage:.1f}% ({i+1}/{total_files})**"
            )
            main_progress.progress(i / total_files)

            current_file_status.markdown(f"**📄 Current: {filename}**")

            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name

            # Process PDF
            documents = process_single_pdf(tmp_path, filename)

            # Clean up
            os.unlink(tmp_path)

            if documents:
                all_documents.extend(documents)
                successful_files += 1
                total_chunks_created += len(documents)

                # Update statistics
                stats_display.markdown(
                    f"""
                **📈 Progress Statistics:**
                - ✅ Files completed: {successful_files}/{total_files}
                - 📄 Total chunks: {total_chunks_created}
                - 📊 Avg chunks/file: {total_chunks_created / successful_files:.1f}
                """
                )

        except Exception as e:
            st.error(f"❌ Error processing {filename}: {str(e)}")

    # Final updates
    main_progress.progress(1.0)
    overall_status.markdown(
        f"**🎉 Complete! {successful_files}/{total_files} files processed**"
    )

    return all_documents


def create_vectorstore(
    documents: List[Document], embedding_model, collection_name: str
):
    """Create vector database with new documents."""
    if not documents:
        return None

    # Create data directory
    os.makedirs("data", exist_ok=True)

    # Show progress
    embedding_status = st.empty()
    embedding_progress = st.progress(0)

    embedding_status.markdown("🔮 **Creating vector embeddings...**")
    embedding_progress.progress(0.3)

    # Initialize Chroma client
    chroma_client = chromadb.PersistentClient(path="data/chroma_db")

    embedding_status.markdown("🔗 **Storing in vector database...**")
    embedding_progress.progress(0.7)

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        client=chroma_client,
        collection_name=collection_name,
        persist_directory="data/chroma_db",
    )

    embedding_progress.progress(1.0)
    embedding_status.markdown("✅ **Vector database created successfully!**")

    return vectorstore


def get_existing_collections() -> List[Dict]:
    """Get list of existing document collections with metadata."""
    try:
        chroma_client = chromadb.PersistentClient(path="data/chroma_db")
        collections = chroma_client.list_collections()

        collection_info = []
        for col in collections:
            try:
                collection = chroma_client.get_collection(col.name)
                count = collection.count()

                # Get sample to count unique files
                sample_result = collection.get(include=["metadatas"], limit=count)
                unique_files = set()
                if sample_result and sample_result.get("metadatas"):
                    for metadata in sample_result["metadatas"]:
                        if metadata and metadata.get("filename"):
                            unique_files.add(metadata["filename"])

                collection_info.append(
                    {
                        "name": col.name,
                        "chunk_count": count,
                        "file_count": len(unique_files),
                        "id": col.id if hasattr(col, "id") else "unknown",
                    }
                )
            except:
                collection_info.append(
                    {
                        "name": col.name,
                        "chunk_count": 0,
                        "file_count": 0,
                        "id": col.id if hasattr(col, "id") else "unknown",
                    }
                )

        return collection_info
    except:
        return []


def connect_to_collection(collection_name: str, embedding_model):
    """Connect to existing collection."""
    try:
        chroma_client = chromadb.PersistentClient(path="data/chroma_db")
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory="data/chroma_db",
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error connecting to collection: {e}")
        return None


def rename_collection(old_name: str, new_name: str) -> bool:
    """Rename a collection by creating a new one and deleting the old one."""
    try:
        # Validate new name
        if not new_name or new_name.strip() == "":
            st.error("Collection name cannot be empty!")
            return False

        new_name = new_name.strip().lower().replace(" ", "_")

        # Check if new name already exists
        existing_collections = get_existing_collections()
        if any(col["name"] == new_name for col in existing_collections):
            st.error(f"Collection '{new_name}' already exists!")
            return False

        # Connect to old collection
        chroma_client = chromadb.PersistentClient(path="data/chroma_db")
        old_collection = chroma_client.get_collection(old_name)

        # Get all documents from old collection
        all_data = old_collection.get(include=["documents", "metadatas", "embeddings"])

        if not all_data["documents"]:
            st.error("No documents found in collection!")
            return False

        # Create new collection with same data
        new_collection = chroma_client.create_collection(new_name)

        # Add all documents to new collection
        new_collection.add(
            documents=all_data["documents"],
            metadatas=all_data["metadatas"],
            embeddings=all_data["embeddings"],
            ids=all_data["ids"],
        )

        # Delete old collection
        chroma_client.delete_collection(old_name)

        return True

    except Exception as e:
        st.error(f"Error renaming collection: {e}")
        return False


def delete_collection(collection_name: str) -> bool:
    """Delete a specific collection."""
    try:
        chroma_client = chromadb.PersistentClient(path="data/chroma_db")
        chroma_client.delete_collection(collection_name)
        return True
    except Exception as e:
        st.error(f"Error deleting collection: {e}")
        return False


# --------------------------------------------------------------
# SEARCH AND CHAT FUNCTIONS
# --------------------------------------------------------------


def search_relevant_documents(query, vectorstore, num_results=15):
    """Search vector database for relevant documents."""
    if not vectorstore:
        return "", []

    results = vectorstore.similarity_search_with_score(query, k=num_results)
    context_parts = []
    search_results = []

    for doc, score in results:
        filename = doc.metadata.get("filename", "unknown")
        page_numbers = doc.metadata.get("page_numbers", "[]")
        title = doc.metadata.get("title", "No Title")

        source_parts = [filename]
        if page_numbers != "[]":
            page_nums = (
                page_numbers.strip("[]").replace("'", "").replace(" ", "").split(",")
            )
            if page_nums and page_nums[0]:
                source_parts.append(f"p. {', '.join(page_nums)}")

        source = f"Source: {' - '.join(source_parts)}"
        if title != "No Title":
            source += f"\nTitle: {title}"

        context_parts.append(f"{doc.page_content}\n{source}")
        search_results.append(
            {
                "content": doc.page_content,
                "filename": filename,
                "title": title,
                "pages": page_numbers,
                "similarity": round(1 - score, 4),
            }
        )

    return "\n\n".join(context_parts), search_results


def generate_chat_response(chat_history, context):
    """Generate chat response using retrieved context."""
    llm, _ = load_models()

    system_prompt = f"""You are a knowledgeable document assistant that answers questions based on the provided documents.
    Use only the information from the context below. If the context doesn't contain relevant information, 
    clearly state this limitation.
    
    Context from documents:
    {context}
    """

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)

    try:
        response_stream = llm.stream(messages)
        full_response = ""
        response_placeholder = st.empty()

        for chunk in response_stream:
            if hasattr(chunk, "content"):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)
        return full_response

    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Please check your GROQ_API_KEY configuration."


# --------------------------------------------------------------
# STREAMLIT APPLICATION
# --------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="Universal Document Assistant", page_icon="📚", layout="wide"
    )

    st.title("📚 Universal Document Assistant")
    st.markdown("*Upload PDFs or entire folders and chat with your documents*")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "upload_session_id" not in st.session_state:
        st.session_state.upload_session_id = hashlib.md5(
            str(time.time()).encode()
        ).hexdigest()[:8]

    # Load models
    llm, embedding_model = load_models()

    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")

        # Upload Mode Selection
        st.subheader("Upload Documents")
        upload_mode = st.radio(
            "Choose upload method:",
            ["📄 Upload PDF Files", "📂 Upload Multiple PDFs from Folder"],
            help="Note: Due to browser limitations, both options require selecting individual files",
        )

        uploaded_files = None
        upload_type = "files"

        if upload_mode == "📄 Upload PDF Files":
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type="pdf",
                accept_multiple_files=True,
                help="Select one or more PDF files individually",
            )
            upload_type = "files"
        else:
            st.info(
                "💡 To upload from a folder: Navigate to your folder, select all PDFs (Ctrl/Cmd+A), then upload"
            )
            uploaded_files = st.file_uploader(
                "Select multiple PDFs from a folder",
                type="pdf",
                accept_multiple_files=True,
                help="Navigate to your folder first, then select all PDF files",
            )
            upload_type = "folder"

        if uploaded_files:
            # Generate smart collection name
            filenames = [f.name for f in uploaded_files]
            suggested_name = generate_smart_collection_name(filenames, upload_type)

            collection_name = st.text_input(
                "Collection Name",
                value=suggested_name,
                help="Smart name generated based on your files",
            )

            # Show preview
            if upload_type == "folder":
                pdf_files = [
                    f for f in uploaded_files if f.name.lower().endswith(".pdf")
                ]
                st.write(f"📄 Found {len(pdf_files)} PDF files:")
                for pdf in pdf_files[:5]:  # Show first 5
                    st.write(f"  • {pdf.name}")
                if len(pdf_files) > 5:
                    st.write(f"  ... and {len(pdf_files) - 5} more")
            else:
                st.write(f"📄 Selected {len(uploaded_files)} PDF files")

            if st.button("🚀 Process Documents", type="primary"):
                with st.expander("📊 Processing Progress", expanded=True):
                    start_time = time.time()
                    st.markdown(
                        f"🕒 **Started:** {pd.Timestamp.now().strftime('%H:%M:%S')}"
                    )

                    documents = process_uploaded_files(
                        uploaded_files, embedding_model, upload_type
                    )

                    if documents:
                        vectorstore = create_vectorstore(
                            documents, embedding_model, collection_name
                        )
                        st.session_state.vectorstore = vectorstore

                        processing_time = time.time() - start_time
                        st.success(
                            f"""
                        🎉 **Processing Complete!**
                        - 📄 Files processed: {len([f for f in uploaded_files if f.name.lower().endswith('.pdf')])}
                        - 📚 Document chunks: {len(documents)}
                        - 🗃️ Collection: {collection_name}
                        - ⏱️ Time: {processing_time:.1f}s
                        """
                        )
                        st.balloons()
                        st.rerun()

        # Load existing collections with rename functionality
        st.divider()
        st.subheader("📚 Existing Collections")

        existing_collections = get_existing_collections()

        if existing_collections:
            for collection in existing_collections:
                # Format display text
                file_text = f"{collection['file_count']} PDF{'s' if collection['file_count'] != 1 else ''}"

                # Create expandable section for each collection
                with st.expander(
                    f"📖 {collection['name']} ({file_text})", expanded=False
                ):
                    # Use responsive columns for buttons
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        if st.button(
                            "📖 Load Collection",
                            key=f"load_{collection['name']}",
                            use_container_width=True,
                        ):
                            vectorstore = connect_to_collection(
                                collection["name"], embedding_model
                            )
                            if vectorstore:
                                st.session_state.vectorstore = vectorstore
                                st.session_state.current_collection_name = collection[
                                    "name"
                                ]
                                st.success(f"✅ Loaded: {collection['name']}")
                                st.rerun()

                    with col2:
                        if st.button(
                            "✏️ Rename",
                            key=f"rename_{collection['name']}",
                            help="Rename collection",
                            use_container_width=True,
                        ):
                            st.session_state.renaming_collection = collection["name"]
                            st.rerun()

                    with col3:
                        if st.button(
                            "🗑️ Delete",
                            key=f"delete_{collection['name']}",
                            help="Delete collection",
                            use_container_width=True,
                            type="secondary",
                        ):
                            if delete_collection(collection["name"]):
                                st.success(f"Deleted {collection['name']}")
                                st.rerun()

                    # Show collection stats
                    stats_col1, stats_col2 = st.columns(2)
                    with stats_col1:
                        st.metric("📄 PDF Files", collection["file_count"])
                    with stats_col2:
                        st.metric("📚 Text Chunks", collection["chunk_count"])

                    # Show rename interface if this collection is being renamed
                    if (
                        hasattr(st.session_state, "renaming_collection")
                        and st.session_state.renaming_collection == collection["name"]
                    ):
                        st.markdown("**✏️ Rename Collection:**")
                        new_name = st.text_input(
                            "New name:",
                            value=collection["name"],
                            key=f"new_name_{collection['name']}",
                        )

                        rename_col1, rename_col2 = st.columns(2)
                        with rename_col1:
                            if st.button(
                                "✅ Save",
                                key=f"save_rename_{collection['name']}",
                                use_container_width=True,
                                type="primary",
                            ):
                                if rename_collection(collection["name"], new_name):
                                    st.success(f"Renamed to: {new_name}")
                                    del st.session_state.renaming_collection
                                    st.rerun()

                        with rename_col2:
                            if st.button(
                                "❌ Cancel",
                                key=f"cancel_rename_{collection['name']}",
                                use_container_width=True,
                            ):
                                del st.session_state.renaming_collection
                                st.rerun()
        else:
            st.info("No collections found. Upload documents to get started!")

        # Current collection info
        if st.session_state.vectorstore:
            st.divider()
            st.subheader("📊 Current Collection")
            try:
                doc_count = st.session_state.vectorstore._collection.count()
                current_name = getattr(
                    st.session_state, "current_collection_name", "Unknown"
                )

                # Get file count for current collection
                try:
                    current_collection = st.session_state.vectorstore._collection
                    sample_result = current_collection.get(
                        include=["metadatas"], limit=doc_count
                    )
                    unique_files = set()
                    if sample_result and sample_result.get("metadatas"):
                        for metadata in sample_result["metadatas"]:
                            if metadata and metadata.get("filename"):
                                unique_files.add(metadata["filename"])
                    file_count = len(unique_files)
                except:
                    file_count = 0

                # Display metrics in columns
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("📄 PDF Files", file_count)
                with metrics_col2:
                    st.metric("📚 Text Chunks", doc_count)
                with metrics_col3:
                    if st.button("✏️ Rename", use_container_width=True):
                        st.session_state.renaming_current = True
                        st.rerun()

                st.write(f"**Collection:** `{current_name}`")

                # Show rename interface for current collection
                if (
                    hasattr(st.session_state, "renaming_current")
                    and st.session_state.renaming_current
                ):
                    st.markdown("**✏️ Rename Current Collection:**")
                    new_name = st.text_input(
                        "New name:", value=current_name, key="current_new_name"
                    )

                    rename_current_col1, rename_current_col2 = st.columns(2)
                    with rename_current_col1:
                        if st.button(
                            "✅ Save",
                            key="save_current_rename",
                            use_container_width=True,
                            type="primary",
                        ):
                            if rename_collection(current_name, new_name):
                                st.success(f"Renamed to: {new_name}")
                                st.session_state.current_collection_name = new_name
                                del st.session_state.renaming_current
                                st.rerun()

                    with rename_current_col2:
                        if st.button(
                            "❌ Cancel",
                            key="cancel_current_rename",
                            use_container_width=True,
                        ):
                            del st.session_state.renaming_current
                            st.rerun()

                if st.button("📄 Clear Current Session", use_container_width=True):
                    st.session_state.vectorstore = None
                    st.session_state.messages = []
                    if hasattr(st.session_state, "current_collection_name"):
                        del st.session_state.current_collection_name
                    st.rerun()

            except:
                st.error("Error reading collection info")

    # Main chat interface
    if not st.session_state.vectorstore:
        st.info("👆 Upload documents using the sidebar to get started!")

        st.markdown(
            """
        ### 📋 How to Use:
        
        **Getting Started:**
        1. **Upload PDFs** - Use the sidebar to select one or multiple PDF files
        2. **Click Process Documents** - Start processing your uploaded files  
        3. **Chat** - Ask questions about your documents once processing is complete
        
        **Collection Management:**
        - Each upload creates a **separate collection** with auto-generated names
        - **Rename** collections to organize your documents better
        - **Delete** collections you no longer need
        - **Switch** between different collections to chat with specific document sets
        - **Load existing collections** from previous sessions
        
        **Chat Features:**
        - Ask questions about document content, summaries, or specific topics
        - Get responses with **source citations** showing which documents and pages contain the information
        - View **relevant document sections** that were used to answer your questions
        """
        )

        return

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_query := st.chat_input("Ask questions about your documents..."):
        # Add user message
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Search and respond
        with st.status("🔍 Searching documents...", expanded=False) as status:
            context, search_results = search_relevant_documents(
                user_query, st.session_state.vectorstore
            )

            if search_results:
                st.write("**📄 Found relevant sections:**")
                for result in search_results:
                    with st.expander(
                        f"📄 {result['filename']} (similarity: {result['similarity']:.3f})"
                    ):
                        st.write(f"**Section:** {result['title']}")
                        st.write(f"**Pages:** {result['pages']}")
                        st.write(result["content"])

        # Generate response
        with st.chat_message("assistant"):
            response = generate_chat_response(st.session_state.messages, context)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
