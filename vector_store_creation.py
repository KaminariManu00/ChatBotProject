import sys
import os
import ollama_model

CHUNK_SIZE = 200
CHUNK_OVERLAP = 10

def process_documents(directory, chunk_size, chunk_overlap):
    combined_content = ""
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        if filepath.endswith(".pdf"):
            with open(filepath, 'rb') as file:
                combined_content += ollama_model.read_pdf(file)
        elif filepath.endswith(".txt"):
            with open(filepath, 'r') as file:
                combined_content += ollama_model.read_txt(file)
        else:
            print(f"File {filename} is not .pdf or .txt")
            continue

    # Split combined content into chunks
    split = ollama_model.split_doc(combined_content, chunk_size, chunk_overlap)

    existing_vector_store = "exam_vector_store"
    create_new = True
    if os.path.exists(f"vector store/{existing_vector_store}"):
        create_new = False

    ollama_model.embedding_storing(split, create_new, existing_vector_store, existing_vector_store)
      

def main():
    directory = sys.argv[1]
    process_documents(directory, CHUNK_SIZE, CHUNK_OVERLAP)

if __name__ == "__main__":
    main()