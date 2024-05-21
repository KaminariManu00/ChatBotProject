import sys
import os
import ollama_model
from docx import Document

# Dimensione del frammento di testo da processare
CHUNK_SIZE = 200
# Sovrapposizione tra frammenti di testo
CHUNK_OVERLAP = 10

# Funzione per leggere un file .docx
def read_docx(file):
    doc = Document(file)
    # Unisce tutti i paragrafi del documento in una stringa unica
    return ' '.join([paragraph.text for paragraph in doc.paragraphs])

# Funzione per processare tutti i documenti in una directory
def process_documents(directory, chunk_size, chunk_overlap):
    combined_content = ""
    
    # Itera su tutti i file nella directory e nelle sue sottodirectory
    for root, dirs, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            
            # Controlla il tipo di file e lo legge di conseguenza
            if filepath.endswith(".pdf"):
                with open(filepath, 'rb') as file:
                    combined_content += ollama_model.read_pdf(file)
            elif filepath.endswith(".txt"):
                with open(filepath, 'r') as file:
                    combined_content += ollama_model.read_txt(file)
            elif filepath.endswith(".docx"):
                with open(filepath, 'rb') as file:
                    combined_content += read_docx(file)
            else:
                print(f"Il file {filename} non è .pdf, .txt o .docx")
                continue

    # Suddivide il contenuto combinato in frammenti
    split = ollama_model.split_doc(combined_content, chunk_size, chunk_overlap)

    existing_vector_store = "exam_vector_store"
    create_new = True
    # Controlla se esiste già un archivio
    if os.path.exists(f"vector store/{existing_vector_store}"):
        create_new = False

    # Memorizza gli embedding dei chunk
    ollama_model.embedding_storing(split, create_new, existing_vector_store, existing_vector_store)
      

def main():
    # Prende la directory da linea di comando
    directory = sys.argv[1]
    process_documents(directory, CHUNK_SIZE, CHUNK_OVERLAP)

if __name__ == "__main__":
    main()