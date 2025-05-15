from langchain_text_splitters import HTMLHeaderTextSplitter, PythonCodeTextSplitter

def chunk_url_content():
    chunks = []
    url = "https://www.york.ac.uk/teaching/cws/wws/webpage1.html"

    html_splitter = HTMLHeaderTextSplitter([("h1", "Header 1")])

    html_header_splits = html_splitter.split_text_from_url(url=url)

    for i, chunk in enumerate(html_header_splits):
        chunks.append({
            "chunk_id": i,
            "chunk": chunk
        })
    
    print(chunks)

if __name__ == '__main__':
    chunk_url_content()