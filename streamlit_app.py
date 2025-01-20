import streamlit as st
import os
import pickle
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage, StorageContext
from llama_index.llms.openai import OpenAI


global prompt, model, indexID, requiredExts

prompt = """
You are an expert at analyzing contracts and your job is to answer technical questions about the contract. 
Assume all questions are related to the provided contracts. 
Keep your answers concise and based on facts – do not hallucinate.""".strip()

model = "gpt-4o-mini"

indexID = "vector_index"

requiredExts = [".pdf", ".txt", "docx" ]

# Show title and description.
st.title("📄 Contract Analysis Agent")

# Get credentials, set up prompt and model
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Set up Settings for LLM and Embeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from llama_index.core import Settings
Settings.llm = OpenAI(model=model, temperature=0, system_prompt=prompt)
Settings.embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")

# Use OpenAI Embedding instead of HuggingFace by uncommenting the following lines
# Settings.embed_model = OpenAIEmbedding(
#     model="text-embedding-3-small", embed_batch_size=100
# )

# Future feature: Add limit to the number of documents to load to speed up the process

@st.cache_resource(show_spinner=False)
def createIndex(directory, flag):
    if not os.path.exists(directory):
        return None
    else:
        reader = SimpleDirectoryReader(
            input_dir=directory, 
            recursive=True, 
            required_exts=requiredExts
        ) 
        docs = reader.load_data()
        files = reader.list_resources()
        if flag == "create":
            index = VectorStoreIndex.from_documents(docs)
            index.set_index_id(indexID)
            index.storage_context.persist(directory)
        elif flag == "load":
            print(f"In Load setting")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=directory)
                print("storage context created")
                index = load_index_from_storage(storage_context, index_id=indexID)
            except:
                print("Index not found. Creating a new one")
                index, files = createIndex(directory, "create")
            else:
                refreshedDocs = index.refresh(docs)
                print(f"Refreshed {sum(refreshedDocs)} documents")
        return index, files



# Get the path to the directory and the index file
directory = "./data"
if not (os.path.exists(directory)):
    directory = st.text_input("Enter directory path:")
if not os.path.isdir(directory):
    st.error("The provided directory path is invalid or is not a directory. Please enter a valid directory path.")
else:
    with st.spinner("Checking to see if contracts have been previously indexed..."):
        index, files = createIndex(directory, "load")   
    if files:
        with st.sidebar:
            st.header("📄 Documents")
            # Diplay the list of documents
            for filePath in files:
                st.write(os.path.basename(filePath))
        st.header("💬 Chat with the Agent")
        # Set up the chat engine
        if "chatEngine" not in st.session_state.keys(): # Initialize the chat engine
            st.session_state.chatEngine = index.as_chat_engine(chat_mode="openai", verbose=True)

        # Initialize chat message history
        if "messages" not in st.session_state.keys(): # Initialize the chat message history
            st.session_state.messages = []
        #         {"role": "assistant", "content": "Who are the parties involved in the contract?"}

        # Display chat messages from history
        if "messages" in st.session_state.keys():
            for message in st.session_state.messages: # Display the prior chat messages
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        # Prompt for user input and save to chat history
        if prompt := st.chat_input("Ask a question about the contract"): # Prompt for user input and save to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chatEngine.chat(prompt)
                    st.markdown(response.response)
                    message = {"role": "assistant", "content": response.response}
                    st.session_state.messages.append(message) # Add response to message history
                    print(st.session_state.messages)

