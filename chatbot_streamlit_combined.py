import streamlit as st
import os
import ollama_model
import torch
import time
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
#progress_bar
#main_placeholder = st.empty()
#def main_place(message="The Task Is Finished !!!!"):
  # main_placeholder.text(message)

# Memory management functions
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    for _ in range(max_retries):
        info = nvmlDeviceGetMemoryInfo(handle)
        if info.free >= min_memory_available:
            break
        print(f"Waiting for {min_memory_available} bytes of free GPU memory. Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    else:
        raise RuntimeError(f"Failed to acquire {min_memory_available} bytes of free GPU memory after {max_retries} retries.")

def main():
    # Call memory management functions before starting Streamlit app
    #min_memory_available = 1 * 1024 * 1024 * 1024  # 1GB
    clear_gpu_memory()
    logo_chat = "/Users/luigibarbato/Desktop/Programmi/Programmi-magistrale/ChatBotProject/logo/logo_chat.png"
    logo_federico_ii = "/Users/luigibarbato/Desktop/Programmi/Programmi-magistrale/ChatBotProject/logo/logo_federico_ii.png"

# Crea due colonne per le immagini
    col1, col2 = st.columns(2)

# Mostra le immagini
    col1.image(logo_chat, width=300, output_format='PNG', use_column_width=False)
    col2.image(logo_federico_ii, width=300, output_format='PNG', use_column_width=False)
    

    # Stili CSS personalizzati per il tema
    st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stButton>button {
        background-color: #f63366;
        color: white;
    }
    .stTitle {
        font-family: 'Roboto', sans-serif;
        font-size: 2.5em;
        color: white;
        text-align: center;
        margin-bottom: 0;
    }
    .stHeader {
        font-family: 'Roboto', sans-serif;
        font-size: 1.5em;
        color: white;
    }
    .stExpander {
        font-family: 'Roboto', sans-serif;
        font-size: 1em;
        color: white;
    }
    .stMarkdown {
        font-family: 'Roboto', sans-serif;
        color: white;
    }
    .stChatMessage {
        background-color: #0E1117;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #cccccc;
    }
    .stChatInput {
        background-color: #0E1117;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='stTitle'>RAG Bot 🤖</h1>", unsafe_allow_html=True)

    # Aggiungi descrizione dell'app
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <p style='font-family: "Roboto", sans-serif; font-size: 1.2em; color: white;'>
            Benvenuto nel Chatbot RAG! Sono l'assistente della legislazione italiana!
        </p>
    </div>
    """, unsafe_allow_html=True)
    #wait_until_enough_gpu_memory()

    display_chatbot_page()
   

def display_chatbot_page():

    st.title("Multi Source Chatbot")

    # Prepare the LLM model
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.session_state.conversation = ollama_model.prepare_rag_llm("llama3", "exam_vector_store")

    # Chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Source documents
    if "source" not in st.session_state:
        st.session_state.source = []

    # Display chats
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Ask a question
    if question := st.chat_input("Ask a question"):
        # Append user question to history
        st.session_state.history.append({"role": "user", "content": question})
        # Add user question
        with st.chat_message("user"):
            st.markdown(question)

        # Answer the question
        answer, doc_source = ollama_model.generate_answer(question)
        with st.chat_message("assistant"):
            st.write(answer)
        # Append assistant answer to history
        st.session_state.history.append({"role": "assistant", "content": answer})

        # Append the document sources
        st.session_state.source.append({"question": question, "answer": answer, "document": doc_source})


    # Source documents
    with st.expander("Chat History and Source Information"):
        st.write(st.session_state.source)



if __name__ == "__main__":
    main()
