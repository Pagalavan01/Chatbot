import sys
import pysqlite3

import os
import streamlit as st
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Try to fix sqlite3 issue for Chroma
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass  # If it fails, we'll try without it

st.set_page_config(page_title="My Personal RAG Chatbot", page_icon="🤖")

# --- OpenRouter / Qwen Setup ---
api_key = st.secrets.get("OPENROUTER_API_KEY")
if not api_key:
    st.error("OPENROUTER_API_KEY not found in secrets.")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# --- Local Embeddings & Chroma Vector Store ---
db_folder = "chroma_db"

if not os.path.exists(db_folder):
    st.error(f"Chroma database not found at '{db_folder}'. Run ingest_local.py first.")
    st.stop()

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = Chroma(persist_directory=db_folder, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 9})

# --- Streamlit UI ---
st.title("🤖 My Personal Chatbot ")
st.caption("Answers only from the Pagalavan's Knowledge base • Powered by Qwen + Local RAG")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about myself..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Retrieve relevant chunks
    with st.spinner("Searching my knowledge base..."):
        docs = retriever.invoke(prompt)
    
    if not docs:
        context = "No relevant information found."
    else:
        context = "\n\n".join([f"From {doc.metadata.get('source', 'unknown')}:\n{doc.page_content}" for doc in docs])

    # Build system prompt with retrieved context
    system_prompt = f"""
You have the data about the Pagalavan. Who ever ask about him answer only from the data.
Answer ONLY using information from the provided context.
Rules:
- Be very brief and to the point, maximum 150 to 250 words.
- Never repeat the same fact twice in one answer.
- If information from different chunks relates, synthesize it naturally — do NOT list everything separately.
- Use at most 3 to 4 bullet points.
- If the question is clearly related to documented info, answer directly — do NOT say "I don't know" prematurely.
- If truly no relevant info exists → say: "I don't have information about that in my knowledge base."
- End with a short friendly note only if it adds value.

Relevant information:
{context}

Be friendly, concise and funny.
"""

    # Prepare messages for Qwen
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model="qwen/qwen3.6-plus:free",
                    messages=api_messages
                )
                bot_response = response.choices[0].message.content
                st.write(bot_response)

                # Append to history only after successful response
                st.session_state.messages.append({"role": "assistant", "content": bot_response})

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Check your OpenRouter key, internet, or rate limits.")

# Sidebar info
with st.sidebar:
    st.header("About This Chatbot")
    st.write("This Chatbot only answer from the Pagalavan's Knowledge base.")
    
    st.divider()
    st.write("Model: qwen/qwen3.6-plus:free (via OpenRouter)")
    st.divider()
    st.write(f"Chunks in knowledge base: **{vectorstore._collection.count()}**")