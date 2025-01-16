import streamlit as st 
from PyPDF2 import PdfReader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain


OPENAI_API_KEY = ""

st.header("My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDf file and start asking question", type="pdf")

# Extract the Text 
if file is not None:
    pdf_reader = PdfReader(file)
    text=""
    for page in pdf_reader.pages:
        text+=page.extract_text()
        # st.write(text)
    #Break it into Chunk 

    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    #st.write(chunks)

    # generating embeddings 
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    #creating vector store - FAISS 

    vector_store = FAISS.from_texts(chunks,embeddings)

    #get user question 
    user_question = st.text_input("Type your question here")

    #do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        # st.write(match)
        # A = question -> user_question
        # B = vector_DB -> vector_store

        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY, 
            temperature = 0, # lower the value reduce the randomness to the answer
            max_token = 1000, # get responses with in 1000 words
            model_name = "gpt-3.5-turbo" # 
        )
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)