from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from htmlTemplates import bot_template, user_template, css

def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_chunk_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=OPENAI_EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL_NAME, temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    system_template  =  """
    Use  the following pieces of context and chat history to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Chat history: {chat_history}

    Question: {question}
    Helpful Answer:
    """
    prompt = PromptTemplate(
        template=system_template,
        input_variables=["context", "question",  "chat_history"],
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        verbose = True,
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return conversation_chain

def handle_user_input(question):
    try: 
        response = st.session_state.conversation({'question': question})
        st.session_state.chat_history = response['chat_history']
    except Exception as e:
        st.error('Please select PDF and click on OK.')

def display_chat_history():
    if st.session_state.chat_history:
        reversed_history = st.session_state.chat_history[::-1]

        formatted_history = []
        for i in range(0, len(reversed_history), 2):
            chat_pair = {
                "AIMessage": reversed_history[i].content,
                "HumanMessage": reversed_history[i + 1].content
            }
            formatted_history.append(chat_pair)

        for i, message in enumerate(formatted_history):
            st.write(user_template.replace("{{MSG}}", message['HumanMessage']), unsafe_allow_html=True)
            st.write(bot_template.replace("{{MSG}}", message['AIMessage']), unsafe_allow_html=True)
  
def main():
    st.set_page_config(page_title='Chat with PDFs', page_icon=':books:')
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('Chat with PDFs :books:')

    question = st.text_input("Ask anything to your PDF:")
    if question:
        handle_user_input(question)

    if st.session_state.chat_history is not None:
        display_chat_history()
      
    with st.sidebar:
        st.subheader("Upload your Documents Here: ")
        pdf_files = st.file_uploader("Choose your PDF Files and Press Process button", type=['pdf'], accept_multiple_files=True)
        
        if pdf_files and st.button("Process"):
            with st.spinner("Processing your PDFs..."):
                try:
                    # Get PDF Text
                    raw_text = get_pdf_text(pdf_files)
                    # Get Text Chunks
                    text_chunks = get_chunk_text(raw_text)
                    # Create Vector Store
                    vector_store = get_vector_store(text_chunks)
                    st.success("Your PDFs have been processed successfully. You can ask questions now.")
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vector_store)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
