import streamlit as st
import os
import ollama_model
import gc

# Funzioni per la gestione della memoria
def clear_gpu_memory():
    gc.collect()

def main():
    # Pulisce la memoria della GPU
    clear_gpu_memory()
    logo_chat = "logo/logo_chat.png"
    logo_federico_ii = "logo/logo_federico_ii.png"

    # Crea due colonne per le immagini
    col1, col2 = st.columns(2)

    # Mostra le immagini
    col1.image(logo_chat, width=300, output_format='PNG', use_column_width=False)
    col2.image(logo_federico_ii, width=300, output_format='PNG', use_column_width=False)
    
    # Stili CSS personalizzati per il tema
    st.markdown("""
    <style>
    .main {
        background-color: #2C3E50;  /* Blu scuro */
        color: #000000;  /* Nero per il testo */
    }
    .stButton>button {
        background-color: #F76C6C;  /* Colore pomodoro per i pulsanti */
        color: white;
    }
    .stTitle {
        font-family: 'Georgia', serif;  /* Font serif elegante */
        font-size: 2.5em;
        color: white;  /* Bianco per il titolo */
        text-align: center;
        margin-bottom: 0;
    }
    .stHeader {
        font-family: 'Georgia', serif;  /* Font serif elegante */
        font-size: 1.5em;
        color: #000000;  /* Nero per gli header */
    }
    .stExpander {
        font-family: 'Georgia', serif;  /* Font serif elegante */
        font-size: 1em;
        color: #000000;  /* Nero per gli espansori */
    }
    .stMarkdown {
        font-family: 'Georgia', serif;  /* Font serif elegante */
        color: #000000;  /* Nero per il markdown */
    }
    .stChatMessage {
        background-color: cadetblue;  /* Blu cadetto */
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #cccccc;  /* Grigio chiaro per il bordo */
    }
    .stChatInput {
        background-color: cadetblue;  /* Blu cadetto */
    }
    .st-emotion-cache-vj1c9o{
        background-color: lightslategray;  /* Grigio ardesia chiaro */
    }
    .st-emotion-cache-1avcm0n{
        background-color: #2C3E50  /* Blu scuro */
    }
    </style>
    """, unsafe_allow_html=True)

    # Titolo dell'app
    st.markdown("<h1 class='stTitle' style='text-align:center;'>RAG Chatbot 🤖</h1>", unsafe_allow_html=True)

    # Descrizione dell'app
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <p style='font-family: "Roboto", sans-serif; font-size: 1.2em; color: white;'>
            Benvenuto nel Chatbot RAG! Sono l'assistente della legislazione italiana!
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Mostra la pagina del chatbot
    display_chatbot_page()
   

def display_chatbot_page():
    # Prepara il modello LLM
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.session_state.conversation = ollama_model.prepare_rag_llm("llama3", "exam_vector_store")

    # Cronologia della chat
    if "history" not in st.session_state:
        st.session_state.history = []

    # Documenti di origine
    if "source" not in st.session_state:
        st.session_state.source = []

    # Mostra le chat
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Fai una domanda
    if question := st.chat_input("Fai una domanda"):
        # Aggiungi la domanda dell'utente alla cronologia
        st.session_state.history.append({"role": "user", "content": question})
        # Aggiungi la domanda dell'utente
        with st.chat_message("user"):
            st.markdown(question)

        # Rispondi alla domanda
        answer, doc_source = ollama_model.generate_answer(question)
        with st.chat_message("assistant"):
            st.write(answer)
        # Aggiungi la risposta dell'assistente alla cronologia
        st.session_state.history.append({"role": "assistant", "content": answer})

        # Aggiungi le fonti del documento
        st.session_state.source.append({"question": question, "answer": answer, "document": doc_source})

    # Documenti di origine
    with st.expander("Cronologia della chat e informazioni sulla fonte"):
        st.write(st.session_state.source)


if __name__ == "__main__":
    main()