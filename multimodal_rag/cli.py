"""Command-line interface for the Multimodal RAG system."""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .rag_system import MultimodalRAG


def main():
    """Run the CLI application."""
    parser = argparse.ArgumentParser(description="Multimodal RAG System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Add document command
    add_parser = subparsers.add_parser("add", help="Add documents to the RAG system")
    add_parser.add_argument("file_paths", nargs="+", help="Path to document files to add")
    add_parser.add_argument(
        "--chunk-size", type=int, default=4000, help="Size of text chunks"
    )
    add_parser.add_argument(
        "--chunk-overlap", type=int, default=200, help="Overlap between chunks"
    )
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("query", nargs="?", help="Query string")
    query_parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    
    # List documents command
    subparsers.add_parser("list", help="List all documents in the system")
    
    # Delete document command
    delete_parser = subparsers.add_parser("delete", help="Delete documents from the system")
    delete_parser.add_argument("doc_ids", nargs="+", help="Document IDs to delete")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize RAG system
    try:
        rag = MultimodalRAG(
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION"),
        )
        rag.initialize()
    except Exception as e:
        print(f"Error initializing RAG system: {e}", file=sys.stderr)
        return 1
    
    # Handle commands
    if args.command == "add":
        for file_path in args.file_paths:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}", file=sys.stderr)
                continue
                
            print(f"Processing {file_path}...")
            try:
                result = rag.add_documents(
                    file_path=file_path,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                )
                print(f"Added {result['total']} elements from {file_path}")
                print(f"- Texts: {result['texts']}")
                print(f"- Tables: {result['tables']}")
                print(f"- Images: {result['images']}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}", file=sys.stderr)
    
    elif args.command == "query":
        import asyncio
        import nest_asyncio
        
        # Apply nest_asyncio to allow nested event loops (useful in Jupyter/notebooks)
        nest_asyncio.apply()
        
        async def run_query():
            try:
                # Initialize the RAG system
                rag = MultimodalRAG()
                await rag.initialize()
                
                if args.interactive:
                    print("Interactive mode. Type 'exit' to quit.")
                    while True:
                        try:
                            query = input("\nEnter your query: ")
                            if query.lower() in ("exit", "quit"):
                                break
                            
                            response = await rag.query(query, return_context=True)
                            print(f"\nAnswer: {response['answer']}")
                            
                            if 'context' in response and isinstance(response['context'], list):
                                print("\nRelevant text snippets:")
                                for i, doc in enumerate(response['context'][:3]):
                                    content = doc.get('content', str(doc))
                                    print(f"{i+1}. {content[:200]}...")
                        
                        except KeyboardInterrupt:
                            print("\nExiting...")
                            break
                        except Exception as e:
                            print(f"Error: {e}", file=sys.stderr)
                            import traceback
                            traceback.print_exc()
                else:
                    if not args.query:
                        print("Error: Query string is required in non-interactive mode", file=sys.stderr)
                        return 1
                        
                    response = await rag.query(args.query, return_context=True)
                    print(response['answer'])
                    
            except Exception as e:
                print(f"Error initializing RAG system: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                return 1
        
        # Create a new event loop for the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async function
            loop.run_until_complete(run_query())
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            # Clean up the event loop
            loop.close()
    
    elif args.command == "list":
        print("List documents functionality not yet implemented.")
        # TODO: Implement document listing
    
    elif args.command == "delete":
        print("Delete documents functionality not yet implemented.")
        # TODO: Implement document deletion
    
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
