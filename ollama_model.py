from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

def read_pdf(file):
    document = ""

    #Estrae il testo per ogni pagina e ritorna il testo concatenando tutte le stringhe del pdf
    reader = PdfReader(file)
    for page in reader.pages:
        document += page.extract_text()

    return document

def read_txt(file):
    #Aumenta la leggibilità del documento, aggiungendo uno spazio precedente ad ogni newline e return
    document = str(file.getvalue())
    document = document.replace("\\n", " \\n ").replace("\\r", " \\r ")

    return document


def split_doc(document, chunk_size, chunk_overlap):

    #vengono settati i parametri presi dall'applicazione finale in streamlit. Il TextSplitter scelto è un RecursiveCharacterTextSplitter
    #Il valore di ritorno sono i chunck del documento originale

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split = splitter.split_text(document)
    split = splitter.create_documents(split)

    return split

def embedding_storing(split, create_new_vs, existing_vector_store, new_vs_name):

    #Funzione usata per gestire e conservare embeddings dei chunks di testo

    #Controllo per sapere se deve essere creato un nuovo vector store
    if create_new_vs is not None:
        #Caricamento dell'instructor degli embedding da HuggingFace
        instructor_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'},
                                           encode_kwargs = {'normalize_embeddings': True})

        #Creazione di un indice FAISS dai chank di testo usando gli embedding caricati
        db = FAISS.from_documents(split, instructor_embeddings)

        if create_new_vs == True:
            #Salva gli indici FAISS creati in una directory locale
            db.save_local("vector store/" + new_vs_name)
        else:
            # Carica indici FAISS già presenti
            load_db = FAISS.load_local(
                "vector store/" + existing_vector_store,
                instructor_embeddings,
                allow_dangerous_deserialization=True
            )
            # Merge dei dui indici
            load_db.merge_from(db)
            load_db.save_local("vector store/" + new_vs_name)


def prepare_rag_llm(llm_model, vector_store_list):
    # Load the embeddings using the HuggingFaceEmbeddings model

    instructor_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'},
                                           encode_kwargs = {'normalize_embeddings': True})

    # Load the FAISS indices from the vector store
    loaded_db = FAISS.load_local(
        f"vector store/{vector_store_list}", instructor_embeddings, allow_dangerous_deserialization=True
    )

    # Load the Ollama model
    local_model = f"{llm_model}"
    llm = ChatOllama(model=local_model)

    # Define the QUERY_PROMPT
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Base the various question asked also on the previous ones.
        Original question: {question}""",
    )

    # Define the retriever
    retriever = MultiQueryRetriever.from_llm(
        loaded_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    # Define the memory
    memory = ConversationBufferWindowMemory(
        k=4,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    # Create the chatbot
    qa_conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,   # Use the retriever defined above
        return_source_documents=True,
        memory=memory,
    )

    # Return the chatbot
    return qa_conversation

def generate_answer(question):
    # Get the history of questions and answers
    history = st.session_state.history

    # Format the history into a string
    history_str = ""
    for qa in history:
        history_str += f"{qa['role']}: {qa['content']}\n"

    # Define a larger prompt that includes the user's question and the history
    larger_prompt = f"""You are an AI language model assistant. Your task is to answer the following question based on the given context and previous questions and answers. Try to provide a comprehensive and accurate answer.
    You have to answer in Italian. Give the tile of the documents you use from the context to answer.
    History:
    {history_str}
    Question: {question}
    """

    answer = "An error has occured"
    doc_source = ["no source"]

    # Invoke the conversation and get the response
    response = st.session_state.conversation.invoke(larger_prompt)
    
    # Extract the answer and the source documents
    answer = response["answer"]
    doc_source = [doc.page_content for doc in response['source_documents']]

    return answer, doc_source
