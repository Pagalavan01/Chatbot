import os
import shutil
from tqdm import tqdm

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma  # This is now from langchain-chroma

# Folders
data_folder = "data"
db_folder = "chroma_db"

# Clear previous DB (optional — remove if you want incremental updates later)
if os.path.exists(db_folder):
    shutil.rmtree(db_folder)
    print("Previous database cleared.")

# Local embedding model (no account needed)
print("Loading embedding model (first run downloads ~300MB, takes 1-2 minutes)...")
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},  # Change to 'cuda' if you have a GPU
    encode_kwargs={'normalize_embeddings': True}
)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

print("Loading and chunking documents from 'data/' folder...")

documents = []
metadatas = []

if not os.path.exists(data_folder):
    raise FileNotFoundError(f"Folder '{data_folder}' not found. Create it and add .txt files.")

for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(data_folder, filename)
        print(f"Processing {filename}...")
        
        loader = TextLoader(filepath, encoding="utf-8")
        docs = loader.load()
        
        chunks = text_splitter.split_documents(docs)
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk.page_content)
            metadatas.append({
                "source": filename,
                "chunk_index": i
            })

print(f"Loaded {len(documents)} chunks.")

# Create Chroma vector store
print("Embedding and saving to Chroma database...")
vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory=db_folder
)

print(f"Success! Database saved to '{db_folder}'")
print(f"Total chunks stored: {vectorstore._collection.count()}")
print("You can now run the Streamlit app — it will use this local database.")