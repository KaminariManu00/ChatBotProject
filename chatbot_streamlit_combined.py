import streamlit as st
import os
import ollama_model
import torch
import time
import gc

# Memory management functions
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


def main():
    clear_gpu_memory()
    logo_chat = "logo/logo_chat.png"
    logo_federico_ii = "logo/logo_federico_ii.png"

# Crea due colonne per le immagini
    col1, col2 = st.columns(2)

# Mostra le immagini
    col1.image(logo_chat, width=300, output_format='PNG', use_column_width=False)
    col2.image(logo_federico_ii, width=300, output_format='PNG', use_column_width=False)
    
    # Stili CSS personalizzati per il tema
    # Stili CSS personalizzati per il tema
    st.markdown("""
    <style>
    .main {
        background-color: #2C3E50;  # Darker blue
        color: #000000;  # Black for text
    }
    .stButton>button {
        background-color: #F76C6C;  # Tomato color for buttons
        color: white;
    }
    .stTitle {
        font-family: 'Georgia', serif;  # Elegant serif font
        font-size: 2.5em;
        color: white;  # Black for title
        text-align: center;
        margin-bottom: 0;
    }
    .stHeader {
        font-family: 'Georgia', serif;  # Elegant serif font
        font-size: 1.5em;
        color: #000000;  # Black for headers
    }
    .stExpander {
        font-family: 'Georgia', serif;  # Elegant serif font
        font-size: 1em;
        color: #000000;  # Black for expanders
    }
    .stMarkdown {
        font-family: 'Georgia', serif;  # Elegant serif font
        color: #000000;  # Black for markdown
    }
    .stChatMessage {
        background-color: cadetblue;  # Darker blue
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #cccccc;
    }
    .stChatInput {
        background-color: cadetblue;  # Darker blue
    }
    .st-emotion-cache-vj1c9o{
        background-color: lightslategray;
    }
    .st-emotion-cache-1avcm0n{
        background-color: #2C3E50
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='stTitle' style='text-align:center;'>RAG Chatbot 🤖</h1>", unsafe_allow_html=True)

    # Aggiungi descrizione dell'app
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <p style='font-family: "Roboto", sans-serif; font-size: 1.2em; color: white;'>
            Benvenuto nel Chatbot RAG! Sono l'assistente della legislazione italiana!
        </p>
    </div>
    """, unsafe_allow_html=True)

    display_chatbot_page()
   

def display_chatbot_page():

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