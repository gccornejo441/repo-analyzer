
import os
from typing import Dict, List

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, PythonCodeTextSplitter, RecursiveCharacterTextSplitter

EXCLUDE_DIRS = ["node_modules", ".git", "build", "__pycache__"]
ALLOWED_EXTS = ['.py', '.md', '.txt', '.rst']


def smart_filter(directory: str, excluded_dirs: List[str], allowed_extensions: List[str] = None) -> List[str]:
    filtered_files: list = []

    for root, dirs, files in os.walk(directory):
        # Removes exluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]

        for file_name in files:
            # root + file name joined
            file_path = os.path.join(root, file_name)

            # if any of excluded dirs in file_path then continue
            if any(excluded in file_path for excluded in excluded_dirs):
                continue

            if allowed_extensions:
                _, ext = os.path.splitext(file_path)
                if ext.lower() not in allowed_extensions:
                    continue

            filtered_files.append(file_path)
    return filtered_files


def verify_user_path(path): return os.path.exists(
    os.path.normpath(os.path.expanduser(path)))


def crawl_local_files(directory: str) -> List[str]:
    """ Process directory """

    if not verify_user_path(directory):
        print("Path DNE!")
    else:
        return smart_filter(directory=directory, excluded_dirs=EXCLUDE_DIRS,
                            allowed_extensions=ALLOWED_EXTS)


def read_paths(file_paths: List[str]) -> List[Dict[str, str]]:
    """
    Reads content from a list of file paths.
    Returns a list of dictionaries with path and content
    """
    contents: List[str] = []
    for file_path in file_paths:
        if not file_path.strip():
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                contents.append({
                    "file_name": file_path,
                    "content": f.read()
                })

        except Exception as e:
            print(f"Failed to read {file_path}: {e}")

    return contents


def chunk_file_content(file_name: str, content: str) -> List[dict]:
    ext = os.path.splitext(file_name)[1].lower()
    chunks = []

    if ext == ".py":
        splitter = PythonCodeTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_text(content)
    elif ext == ".md":
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")]
        )
        split_docs = splitter.split_text(content)
        splits = [doc.page_content for doc in split_docs]
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50)
        splits = splitter.split_text(content)

    for i, chunk in enumerate(splits):
        chunks.append({
            "file_name": file_name,
            "chunk_id": i,
            "chunk": chunk
        })

    return chunks


def chunk_codebase_documents(file_data: List[dict]) -> List[dict]:
    """
    Takes list of {"file_name", "content"} and returns chunked documents.
    """
    all_chunks = []
    for doc in file_data:
        chunks = chunk_file_content(doc["file_name"], doc["content"])
        all_chunks.extend(chunks)
    return all_chunks


def generate_embeddings(chunked_docs: List[dict]):
    documents = [
        Document(
            page_content=chunk["chunk"],
            metadata={
                "file_name": chunk["file_name"],
                "chunk_id": chunk["chunk_id"]
            }
        )
        for chunk in chunked_docs
    ]

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

    texts = [doc.page_content for doc in documents]
    embeddings = embedding_model.embed_documents(texts)

    print(f"Generated {len(embeddings)} embeddings")
    return embeddings, documents


def save_to_vector_store(
        documents: List[Document],
        embedding_model: HuggingFaceEmbeddings,
        persist_path="repo_index"):

    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(persist_path)
    print(f"Save vector store to '{persist_path}'")
    return vectorstore


def search_codebase(vectorstore, query: str, k: int = 5):
    results = vectorstore.similarity_search(query, k=k)
    for i, doc in enumerate(results):
        print(
            f"\n[Match {i+1}] - {doc.metadata.get('file_name')} (chunk {doc.metadata.get('chunk_id')})")
        print(doc.page_content)


if __name__ == '__main__':
    from_user = input("Enter directory path: ")
    filtered_files = crawl_local_files(from_user)
    file_contents = read_paths(filtered_files)
    chunked_docs = chunk_codebase_documents(file_contents)
    embedding, documents = generate_embeddings(chunked_docs)
    vectorstore = save_to_vector_store(documents, HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"))
    search_codebase(vectorstore, "How is authentication handled?")
