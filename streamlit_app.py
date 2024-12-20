import streamlit as st
import os
import pickle
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from CustomDirectoryReader import CustomDirectoryReader

# Show title and description.
st.title("📄 Contract Analysis Agent")
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
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from llama_index.core import Settings
Settings.llm = OpenAI(model=model, temperature=0, system_prompt=prompt)
Settings.embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")

# Use OpenAI Embedding instead of HuggingFace by uncommenting the following lines
# Settings.embed_model = OpenAIEmbedding(
#     model="text-embedding-3-small", embed_batch_size=100
# )




# Load the data and create the index
# @st.cache_resource(show_spinner=False)
# def loadData(inputDir = "./data"):
#     with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
#         reader = SimpleDirectoryReader(input_dir=inputDir, recursive=True)
#         docs = reader.load_data()
#         index = VectorStoreIndex.from_documents(docs)
#         return index

# Future feature: Add limit to the number of documents to load to speed up the process

@st.cache_resource(show_spinner=False)
def refresh_index(directory, indexFile='index.pkl'):
    reader = SimpleDirectoryReader(
        input_dir=directory, 
        recursive=True, 
        exclude=os.path.join(directory, indexFile)
    )
    # reader = CustomDirectoryReader(directory, exclude_files=[indexFile], recursive=True)
    docs, files = reader.load_data()
    index = VectorStoreIndex.from_documents(docs)
    indexFilePath = os.path.join(directory, indexFile)
    index.save_to_disk(indexFilePath)
    return index, files

def load_index(directory, indexFile='index.pkl'):
    indexFilePath = os.path.join(directory, indexFile)
    if os.path.exists(indexFilePath):
        index = VectorStoreIndex.load_from_disk(indexFilePath)
        #Get the files in the folder
        reader = CustomDirectoryReader(directory, exclude_files=[indexFile], recursive=True)
        _, files = reader.load_data()
        return index, files
    else:
        return None, []



# Get the path to the directory and the index file
directory = st.text_input("Enter directory path:")
indexFile = 'index.pkl'
if st.button("Testing"):
    directory = "c:/Users/ryanm/OneDrive"
    st.write("Testing button clicked!")
    # save index to disk
    st.write(f"Directory {directory}")
    reader = SimpleDirectoryReader(
        input_dir=directory, 
        #recursive=True, 
        exclude=os.path.join(directory, indexFile),
        num_files_limit=2
    )
    docs = reader.load_data()
    index = VectorStoreIndex.from_documents(docs)
    index.set_index_id("vector_index")
    index.storage_context.persist(directory)
    
# Call load data function
if directory and st.button("Continue"):
    #index = loadData(directory)
    if st.button("Refresh Index"):
        index, files = refresh_index(directory, indexFile)
        st.write("Index refreshed!")
    else:
        index, files = load_index(indexFile)
        if index:
            st.write("Index loaded from file.")
        else:
            st.write("No index found. Please refresh the index.")
    if files:
        # Split the main frame into two columns for the chat and the contract
        col1, col2 = st.columns([0.3, 0.7], vertical_alignment="center")

        with col1: 
            st.header("📄 Documents")
            # Diplay the list of documents
            for filePath in files:
                st.write(os.path.basename(filePath))
            # for fileResource in reader.list_resources():
            #     st.write(os.path.basename(fileResource))

        with col2:
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

