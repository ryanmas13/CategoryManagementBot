import streamlit as st

import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# Show title and description.
st.title("📄 Document question answering ")
# st.write(
#     "Upload your contract below and ask a question about it "
# )

# Get credentials, set up prompt and model
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

prompt = """
You are an expert at analyzing contracts and your job is to answer technical questions about the contract. 
Assume all questions are related to the provided contracts. 
Keep your answers concise and based on facts – do not hallucinate.""".strip()

model = "gpt-4o-mini"

# Set up Settings for LLM and Embeddings
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from llama_index.core import Settings
Settings.llm = OpenAI(model=model, temperature=0, system_prompt=prompt)
Settings.embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")

# Use OpenAI Embedding instead of HuggingFace by uncommenting the following lines
# Settings.embed_model = OpenAIEmbedding(
#     model="text-embedding-3-small", embed_batch_size=100
# )

# Setup chat message history
if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Who are the parties involved in the contract?"},
    ]


# Load the data and create the index
@st.cache_resource(show_spinner=False)
def loadData():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        # service_context = ServiceContext.from_defaults(
        #     llm=OpenAI(
        #             model=model,
        #             temperature=0, 
        #             system_prompt=prompt
        #             )
        #     )
        index = VectorStoreIndex.from_documents(docs)
        return index

# Call load data function
index = loadData()


# Set up the chat engine
if "chatEngine" not in st.session_state.keys(): # Initialize the chat engine
    st.session_state.chatEngine = index.create_chat_engine(chat_mode="openai", llm=model, verbose=True)



# Prompt for user input and save to chat history
if prompt := st.chat_input("Ask a question about the contract"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])


# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chatEngine.stream_chat(prompt)
            st.write_stream(response.response_gen)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history

# # Split the provided document into chunks
# from langchain.text_splitter import CharacterTextSplitter

# def split_text(file):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size= 500,
#         chunk_overlap = 20, 
#         length_function = len
#     )
#     chunks = text_splitter.split_documents(file)
#     return chunks

# # Setup the LLM
# # Add code here to set up the IBM LLM or OpenAI LL

# # Define the prompt
# from langchain import hub
# prompt = hub.pull("rlm/rag-prompt")



# # Let the user upload a file via `st.file_uploader`.
# uploaded_pdf = st.file_uploader(
#     "Upload a document (.pdf)", type=("pdf")
# )

# # For testing purposes, we will use a dummy pdf

    
# # Ask the user for a question via `st.text_area`.
# question = st.text_area(
#     "Now ask a question about the document!",
#     placeholder="Can you give me a short summary of this contract?",
#     disabled=not uploaded_pdf,
# )

# from PyPDF2 import PdfReader
# from langchain_ibm import WatsonxEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough


# if uploaded_pdf and question:
#     st.write("Here")
#     pdf_reader = PdfReader(uploaded_pdf)
#     text = ""
#     for page in pdf_reader.pages:
#       text += page.extract_text()
#     st.write("Splitting text")
#     #Split the incoming text
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size= 500,
#         chunk_overlap = 20, 
#         length_function = len
#     )
#     chunks = text_splitter.split_text(text)
#     #Create embeddings
#     st.write("Creating embeddings")
#     embeddings = WatsonxEmbeddings(
#     model_id="ibm/slate-30m-english-rtrvr",
#     url=credentials["url"],
#     apikey=credentials["apikey"],
#     project_id=projectID
#     )
#     #Setup vectorstore using FAISS
#     ids = [str(i) for i in range(0, len(chunks))]
#     faissdb = FAISS.from_texts(chunks, embeddings, ids=ids)
#     #Setup retreiver
#     retriever = faissdb.as_retriever()
#     st.write("Setup the chain")
#     llm = LLM()
#     rag_chain = (
#             {
#                 "context": retriever,
#                 "question": RunnablePassthrough()
#             }
#             | prompt
#             | llm
#             | StrOutputParser()
#         )
#     response = rag_chain.invoke(question)
#     st.text(response)


    