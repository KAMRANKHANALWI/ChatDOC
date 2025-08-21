# enhanced_search_interface.py
# Command-line search interface with collection management

import os
import argparse
import warnings
import chromadb
from typing import List, Dict
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Configure environment
warnings.filterwarnings("ignore", message=".*pin_memory.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_DB_PATH = "data/chroma_db"


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Search and interact with your document collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_search_interface.py                    # List available collections
  python enhanced_search_interface.py -c documents      # Search in 'documents' collection
  python enhanced_search_interface.py -c my_pdfs -q "machine learning"  # Direct search
  python enhanced_search_interface.py --list            # List all collections
  python enhanced_search_interface.py --delete my_collection  # Delete collection
        """,
    )

    parser.add_argument(
        "-c",
        "--collection",
        help="Collection name to search in",
    )

    parser.add_argument(
        "-q",
        "--query",
        help="Search query (if not provided, interactive mode starts)",
    )

    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )

    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Path to vector database (default: {DEFAULT_DB_PATH})",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available collections",
    )

    parser.add_argument(
        "--delete",
        help="Delete specified collection",
    )

    parser.add_argument(
        "--chat",
        action="store_true",
        help="Enable chat mode with AI responses",
    )

    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Show detailed metadata for search results",
    )

    return parser.parse_args()


def get_available_collections(db_path: str) -> List[Dict]:
    """Get list of available collections with metadata."""
    try:
        if not os.path.exists(db_path):
            return []

        chroma_client = chromadb.PersistentClient(path=db_path)
        collections = chroma_client.list_collections()

        collection_info = []
        for col in collections:
            try:
                collection = chroma_client.get_collection(col.name)
                count = collection.count()

                # Try to get sample metadata
                sample_result = collection.peek(limit=1)
                sample_metadata = {}
                if (
                    sample_result
                    and sample_result.get("metadatas")
                    and len(sample_result["metadatas"]) > 0
                ):
                    sample_metadata = sample_result["metadatas"][0]

                collection_info.append(
                    {
                        "name": col.name,
                        "count": count,
                        "id": col.id if hasattr(col, "id") else "unknown",
                        "sample_metadata": sample_metadata,
                    }
                )
            except Exception as e:
                collection_info.append(
                    {
                        "name": col.name,
                        "count": 0,
                        "id": "unknown",
                        "sample_metadata": {},
                        "error": str(e),
                    }
                )

        return collection_info
    except Exception as e:
        print(f"❌ Error accessing database: {e}")
        return []


def list_collections(db_path: str):
    """List all available collections."""
    print("📚 Available Document Collections:")
    print("=" * 50)

    collections = get_available_collections(db_path)

    if not collections:
        print("No collections found.")
        print(f"💡 Database path: {db_path}")
        print("💡 Use enhanced_pdf_processor.py to create collections")
        return

    for i, col in enumerate(collections, 1):
        print(f"\n{i}. 📖 {col['name']}")
        print(f"   📄 Documents: {col['count']}")

        if col.get("sample_metadata"):
            metadata = col["sample_metadata"]
            if metadata.get("filename"):
                print(f"   📝 Example file: {metadata['filename']}")
            if metadata.get("source_type"):
                print(f"   📂 Source type: {metadata['source_type']}")

        if col.get("error"):
            print(f"   ⚠️  Warning: {col['error']}")


def delete_collection(collection_name: str, db_path: str) -> bool:
    """Delete a specific collection."""
    try:
        chroma_client = chromadb.PersistentClient(path=db_path)

        # Check if collection exists
        collections = chroma_client.list_collections()
        if not any(col.name == collection_name for col in collections):
            print(f"❌ Collection '{collection_name}' not found!")
            return False

        # Confirm deletion
        print(f"⚠️  Are you sure you want to delete collection '{collection_name}'?")
        confirmation = input("Type 'yes' to confirm: ").strip().lower()

        if confirmation == "yes":
            chroma_client.delete_collection(collection_name)
            print(f"✅ Collection '{collection_name}' deleted successfully!")
            return True
        else:
            print("❌ Deletion cancelled.")
            return False

    except Exception as e:
        print(f"❌ Error deleting collection: {e}")
        return False


def connect_to_collection(collection_name: str, db_path: str):
    """Connect to a specific collection."""
    try:
        if not os.path.exists(db_path):
            print(f"❌ Database not found at: {db_path}")
            return None

        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        chroma_client = chromadb.PersistentClient(path=db_path)

        vectorstore = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=db_path,
        )

        # Test connection
        count = vectorstore._collection.count()
        print(f"✅ Connected to '{collection_name}' ({count} documents)")

        return vectorstore

    except Exception as e:
        print(f"❌ Error connecting to collection '{collection_name}': {e}")
        return None


def search_documents(vectorstore, query: str, k: int = 5, show_metadata: bool = False):
    """Search documents and display results."""
    try:
        print(f"\n🔍 Searching for: '{query}'")
        print("=" * 50)

        results = vectorstore.similarity_search_with_score(query, k=k)

        if not results:
            print("No results found.")
            return []

        search_results = []
        for i, (doc, score) in enumerate(results, 1):
            filename = doc.metadata.get("filename", "unknown")
            title = doc.metadata.get("title", "No Title")
            page_numbers = doc.metadata.get("page_numbers", "[]")
            similarity = round(1 - score, 4)

            print(f"\n📄 Result {i} (Similarity: {similarity:.3f})")
            print(f"📁 File: {filename}")
            print(f"📝 Title: {title}")

            if page_numbers != "[]":
                page_nums = (
                    page_numbers.strip("[]")
                    .replace("'", "")
                    .replace(" ", "")
                    .split(",")
                )
                if page_nums and page_nums[0]:
                    print(f"📖 Pages: {', '.join(page_nums)}")

            print(f"📄 Content Preview:")
            content_preview = (
                doc.page_content[:300] + "..."
                if len(doc.page_content) > 300
                else doc.page_content
            )
            print(f"   {content_preview}")

            if show_metadata:
                print(f"🔍 Metadata:")
                for key, value in doc.metadata.items():
                    if key not in ["filename", "title", "page_numbers"]:
                        print(f"   {key}: {value}")

            search_results.append(
                {
                    "content": doc.page_content,
                    "filename": filename,
                    "title": title,
                    "pages": page_numbers,
                    "similarity": similarity,
                    "metadata": doc.metadata,
                }
            )

        return search_results

    except Exception as e:
        print(f"❌ Search error: {e}")
        return []


def generate_ai_response(query: str, search_results: List[Dict]) -> str:
    """Generate AI response based on search results."""
    try:
        llm = ChatGroq(
            model="llama3-8b-8192", temperature=0.1, api_key=os.getenv("GROQ_API_KEY")
        )

        # Prepare context from search results
        context_parts = []
        for result in search_results:
            source = f"Source: {result['filename']}"
            if result["pages"] != "[]":
                source += f" - {result['pages']}"

            context_parts.append(f"{result['content']}\n{source}")

        context = "\n\n".join(context_parts)

        system_prompt = f"""You are a knowledgeable document assistant. Answer the user's question based only on the provided context from the documents. If the context doesn't contain relevant information, clearly state this limitation.

Context from documents:
{context}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        return f"❌ Error generating AI response: {e}"


def interactive_search(
    vectorstore, chat_mode: bool = False, show_metadata: bool = False
):
    """Interactive search mode."""
    print("\n🎯 Interactive Search Mode")
    print("Type 'quit' or 'exit' to stop, 'help' for commands")
    print("=" * 50)

    while True:
        try:
            query = input("\n🔍 Enter search query: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if query.lower() == "help":
                print(
                    """
Available commands:
  help     - Show this help message
  quit     - Exit the program
  exit     - Exit the program
  
Just type your search query to search the documents.
                """
                )
                continue

            if not query:
                print("Please enter a search query.")
                continue

            # Perform search
            search_results = search_documents(
                vectorstore, query, k=5, show_metadata=show_metadata
            )

            # Generate AI response if chat mode is enabled
            if chat_mode and search_results:
                print(f"\n🤖 AI Response:")
                print("-" * 30)
                ai_response = generate_ai_response(query, search_results)
                print(ai_response)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function."""
    args = parse_arguments()

    print("🔍 Enhanced Document Search Interface")
    print("=" * 50)

    # Handle list collections
    if args.list:
        list_collections(args.db_path)
        return

    # Handle delete collection
    if args.delete:
        delete_collection(args.delete, args.db_path)
        return

    # Show available collections if no specific collection specified
    if not args.collection:
        print("📚 Available collections:")
        collections = get_available_collections(args.db_path)

        if not collections:
            print("No collections found.")
            print("\n💡 Tips:")
            print("   - Use enhanced_pdf_processor.py to create collections")
            print("   - Check database path with --db-path")
            return

        for i, col in enumerate(collections, 1):
            print(f"   {i}. {col['name']} ({col['count']} documents)")

        print(f"\n💡 Use -c <collection_name> to search in a specific collection")
        print(f"💡 Use --list for detailed collection info")
        return

    # Connect to collection
    vectorstore = connect_to_collection(args.collection, args.db_path)
    if not vectorstore:
        return

    # Handle direct query
    if args.query:
        search_results = search_documents(
            vectorstore, args.query, k=args.top_k, show_metadata=args.show_metadata
        )

        # Generate AI response if chat mode is enabled
        if args.chat and search_results:
            print(f"\n🤖 AI Response:")
            print("-" * 30)
            ai_response = generate_ai_response(args.query, search_results)
            print(ai_response)
    else:
        # Interactive mode
        interactive_search(vectorstore, args.chat, args.show_metadata)


if __name__ == "__main__":
    main()
