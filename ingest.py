"""Build the RAG index: scheme markdown files -> section chunks -> embeddings -> FAISS.

Run once (and again whenever you edit data/schemes/*.md):
    python ingest.py
"""
from config import SCHEMES_DIR, VECTORSTORE_DIR, get_embeddings
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "scheme"), ("##", "section")], strip_headers=True
)
size_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)


def load_chunks() -> list[Document]:
    chunks = []
    for path in sorted(SCHEMES_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for part in header_splitter.split_text(text):
            scheme = part.metadata.get("scheme", path.stem)
            section = part.metadata.get("section", "Overview")
            for piece in size_splitter.split_text(part.page_content):
                # Prefix every chunk with its scheme and section so retrieval
                # and the LLM always know which scheme a chunk belongs to.
                chunks.append(Document(
                    page_content=f"Scheme: {scheme}\nSection: {section}\n{piece}",
                    metadata={"source": path.name, "scheme_id": path.stem,
                              "scheme": scheme, "section": section},
                ))
    return chunks


def main():
    chunks = load_chunks()
    db = FAISS.from_documents(chunks, get_embeddings())
    db.save_local(str(VECTORSTORE_DIR))
    n_files = len({c.metadata["source"] for c in chunks})
    print(f"Indexed {n_files} scheme files as {len(chunks)} chunks -> {VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
