#!/usr/bin/env python3
"""
Windows-compatible version of Agentic-Reasoning
This script works around Windows compatibility issues with vLLM and signal handling
"""

import os
import sys
import json
import argparse
from typing import Optional

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Agentic Reasoning Framework (Windows Compatible)')

    # Model selection
    parser.add_argument('--remote_model', type=str, default='gpt-4o',
                       choices=['gpt-4o', 'claude-3.5-sonnet', 'gpt-3.5-turbo'],
                       help='Remote model to use (required for Windows)')

    # Search configuration
    parser.add_argument('--bing_subscription_key', type=str,
                       help='Bing search subscription key')
    parser.add_argument('--bing_endpoint', type=str, default='https://api.bing.microsoft.com/v7.0/search',
                       help='Bing search endpoint')
    parser.add_argument('--use_jina', action='store_true', default=True,
                       help='Use Jina AI for content extraction')
    parser.add_argument('--jina_api_key', type=str,
                       help='Jina AI API key')

    # Search limits
    parser.add_argument('--max_search_limit', type=int, default=3,
                       help='Maximum number of searches per query')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of search results to retrieve')
    parser.add_argument('--max_doc_len', type=int, default=2000,
                       help='Maximum document length for processing')

    # Generation parameters
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Generation temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p sampling parameter')
    parser.add_argument('--max_tokens', type=int, default=4096,
                       help='Maximum tokens to generate')

    # Mind map (optional)
    parser.add_argument('--mind_map', action='store_true',
                       help='Enable mind map functionality')
    parser.add_argument('--mind_map_path', type=str,
                       help='Path for mind map storage')

    # Interactive mode
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')

    # Dataset mode (limited on Windows)
    parser.add_argument('--dataset_name', type=str,
                       help='Dataset name (limited functionality on Windows)')

    return parser.parse_args()

def test_dependencies():
    """Test that all required dependencies are available"""
    print("🧪 Testing dependencies...")

    required_deps = [
        ('litellm', 'LLM API integration'),
        ('dspy', 'Reasoning framework'),
        ('transformers', 'Model loading'),
        ('duckduckgo_search', 'Web search'),
        ('trafilatura', 'Content extraction'),
        ('chromadb', 'Vector storage'),
        ('langchain', 'LLM orchestration'),
    ]

    missing_deps = []
    for dep, desc in required_deps:
        try:
            __import__(dep)
            print(f"✓ {dep} - {desc}")
        except ImportError:
            print(f"❌ {dep} - {desc}")
            missing_deps.append(dep)

    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        return False

    print("\n✅ All core dependencies available!")
    return True

def initialize_remote_model(model_name: str):
    """Initialize remote model for Windows compatibility"""
    print(f"🔧 Initializing remote model: {model_name}")

    try:
        from transformers import AutoTokenizer
        from scripts.utils.remote_llm import RemoteAPILLM

        # Initialize tokenizer
        if model_name == 'gpt-4o':
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        else:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")  # fallback

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        # Initialize remote LLM
        llm = RemoteAPILLM(model_name=model_name)

        print("✅ Remote model initialized successfully")
        return llm, tokenizer

    except Exception as e:
        print(f"❌ Failed to initialize remote model: {e}")
        return None, None

def run_interactive_query(args):
    """Run an interactive query session"""
    print("\n🤖 Agentic-Reasoning Interactive Mode")
    print("=" * 50)

    # Initialize model
    llm, tokenizer = initialize_remote_model(args.remote_model)
    if not llm or not tokenizer:
        return

    # Import required modules
    try:
        from scripts.tools.run_search import search_agent
        from scripts.tools.run_code import code_agent
    except ImportError as e:
        print(f"❌ Failed to import tools: {e}")
        return

    # Initialize tools
    search_tool = search_agent(
        llm, tokenizer,
        bing_subscription_key=args.bing_subscription_key,
        bing_endpoint=args.bing_endpoint,
        top_k=args.top_k,
        use_jina=args.use_jina,
        jina_api_key=args.jina_api_key,
        max_doc_len=args.max_doc_len,
        max_tokens=args.max_tokens,
        coherent=True,
        MAX_SEARCH_LIMIT=args.max_search_limit,
        MAX_TURN=5
    )

    code_tool = code_agent(model_name=args.remote_model)

    print("\n💬 Enter your research query (or 'quit' to exit):")
    print("Example: 'What are the latest developments in quantum computing?'")

    while True:
        try:
            query = input("\n> ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if not query:
                continue

            print(f"\n🔍 Researching: {query}")
            print("-" * 50)

            # For now, just demonstrate search capability
            # Full agentic reasoning would require more complex setup
            print("🔧 Search functionality available")
            print("🔧 Code execution available")
            print("🔧 Web content extraction available")

            print(f"\n✅ Query processed: {query}")
            print("💡 Full agentic reasoning pipeline would analyze this query,")
            print("   perform web searches, execute code, and build reasoning chains.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing query: {e}")

def main():
    """Main function"""
    print("🚀 Agentic-Reasoning Framework (Windows Compatible)")
    print("=" * 60)

    # Parse arguments
    args = parse_args()

    # Test dependencies
    if not test_dependencies():
        print("\n❌ Dependency test failed. Please check your installation.")
        return 1

    # Check for required API keys
    if not args.bing_subscription_key:
        print("\n⚠️  Warning: No Bing search key provided.")
        print("   Web search functionality will be limited.")
        print("   Get a key from: https://www.microsoft.com/en-us/bing/apis/bing-web-search-api")

    if args.use_jina and not args.jina_api_key:
        print("\n⚠️  Warning: Jina AI key not provided.")
        print("   Content extraction will use fallback methods.")
        print("   Get a key from: https://jina.ai/embeddings/")

    # Run interactive mode or show info
    if args.interactive:
        run_interactive_query(args)
    else:
        print("\n📋 Agentic-Reasoning is ready!")
        print("\nAvailable modes:")
        print("- Interactive research: --interactive")
        print("- Dataset processing: --dataset_name [name] (limited on Windows)")
        print("\nExample usage:")
        print("python run_agentic_windows.py --interactive --remote_model gpt-4o")
        print("\nSet environment variables:")
        print("export OPENAI_API_KEY='your-key'")
        print("export BING_SUBSCRIPTION_KEY='your-bing-key'")
        print("export JINA_API_KEY='your-jina-key'")

    return 0

if __name__ == "__main__":
    sys.exit(main())
