import sys
import argparse
from pathlib import Path

from . import config
from .utils import load_resumes, chunk_text, build_corpus_stats, embed_chunks, get_query_embedding
from .vector_store import HybridStore
from .model_wrapper import LLM

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
DIM    = "\033[2m"


def print_banner(model_name: str):
    print(f"{BOLD}{CYAN}")
    print("╔══════════════════════════════════════════╗")
    print("║             Resume-RAG  v2.0             ║")
    print("║       BM25 + TF-IDF  ·  Hybrid Search    ║")
    print("╚══════════════════════════════════════════╝")
    print(f"{DIM}Model: {BOLD}{model_name}{RESET}\n")


def build_index(force: bool = False) -> HybridStore:
    store = HybridStore()

    if config.INDEX_FILE.exists() and not force:
        print(f"{DIM}Loading cached index from {config.INDEX_FILE.name}…{RESET}")
        store.load(config.INDEX_FILE)
        print(f"{GREEN}[✓] Loaded {len(store)} chunks from cache.{RESET}\n")
        return store

    print(f"{DIM}Building index from {config.DATA_DIR}…{RESET}")
    docs = load_resumes(config.DATA_DIR)
    if not docs:
        print("[!] No .txt resumes found in data/")
        sys.exit(1)

    chunks, ids, sources = [], [], []
    for doc_id, text in docs.items():
        for i, chunk in enumerate(chunk_text(text)):
            chunks.append(chunk)
            ids.append(f"{doc_id}#c{i}")
            sources.append(doc_id)

    vocab, doc_freq = build_corpus_stats(chunks)
    vectors = embed_chunks(chunks, vocab, doc_freq)

    store.vocab = vocab
    store.doc_freq = doc_freq
    store.add(chunks, vectors, ids)
    store.save(config.INDEX_FILE)

    print(f"{GREEN}[✓] Indexed {len(chunks)} chunks from {len(docs)} resume(s): "
          f"{', '.join(docs.keys())}{RESET}\n")
    return store


def answer_query(query: str, store: HybridStore, llm: LLM) -> None:
    query_vec = get_query_embedding(query, store.vocab, store.doc_freq, len(store))
    results = store.search(query, query_vec, k=config.TOP_K)

    if not results:
        print(f"{YELLOW}No results found.{RESET}\n")
        return

    context_chunks = [f"[Candidate: {cid.split('#')[0]}]\n{chunk}" for cid, _, chunk in results]

    print(f"\n{BOLD}{GREEN}Answer:{RESET}")
    print("  ", end="", flush=True)
    for token in llm.generate_stream("", context_chunks=context_chunks, query=query):
        print(token, end="", flush=True)
    print("\n")

    print(f"{DIM}Sources:{RESET}")
    seen = set()
    for cid, score, _ in results:
        doc = cid.split("#")[0]
        if doc not in seen:
            print(f"  {DIM}· {doc}  (score: {score:.3f}){RESET}")
            seen.add(doc)
    print()


def print_help():
    print(f"{DIM}Commands:")
    print("  /list          - Show all indexed resumes")
    print("  /reindex       - Rebuild vector + BM25 index")
    print("  /model [name]  - View or switch active Ollama model")
    print("  /help          - Show this help")
    print(f"  exit           - Quit{RESET}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume-RAG")
    parser.add_argument("--reindex", action="store_true", help="Force rebuild the index")
    parser.add_argument("--model", type=str, default=None, help="Ollama model name (e.g. llama3.1:8b, qwen2.5:7b)")
    args = parser.parse_args()

    active_model = args.model or config.MODEL_NAME
    llm = LLM(model=active_model)

    print_banner(llm.model)
    store = build_index(force=args.reindex)

    print(f"{BOLD}Ask a question about the resumes.{RESET}  Type {DIM}/help{RESET} for commands.\n")

    while True:
        try:
            query = input(f"{CYAN}❯ {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if query == "/help":
            print_help()
            continue
        if query == "/list":
            docs = load_resumes(config.DATA_DIR)
            print(f"{DIM}Indexed resumes: {', '.join(docs.keys())}{RESET}\n")
            continue
        if query == "/reindex":
            store = build_index(force=True)
            continue
        if query.startswith("/model"):
            parts = query.split(maxsplit=1)
            if len(parts) == 1:
                print(f"{DIM}Current active model: {BOLD}{llm.model}{RESET}\n")
            else:
                new_model = parts[1].strip()
                llm.model = new_model
                print(f"{GREEN}[✓] Switched model to: {BOLD}{new_model}{RESET}\n")
            continue

        answer_query(query, store, llm)


if __name__ == "__main__":
    main()
