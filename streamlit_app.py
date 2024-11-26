import streamlit as st
#from openai import OpenAI

# Show title and description.
st.title("📄 Document question answering")
st.write(
    "Upload your contract below and ask a question about it "
)

# Get credentials

apikey = st.secrets.IBM.API_KEY
url = st.secrets.IBM.URL

from ibm_watsonx_ai import Credentials

credentials = Credentials(
    url = url,
    api_key = apikey
)

try:
    projectID = st.secrets.IBM.PROJECT_ID
except KeyError:
    projectID = st.text_input("Couldn't find project ID to run this project. Please contact the developer")

from ibm_watsonx_ai import APIClient
WatsonXAI = APIClient(credentials, projectID)

# Split the provided document into chunks
from langchain.text_splitter import CharacterTextSplitter

def split_text(file):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size= 500,
        chunk_overlap = 20, 
        length_function = len
    )
    chunks = text_splitter.split_documents(file)
    return chunks

# Setup the LLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from langchain_ibm import WatsonxLLM

def LLM():

    model_id = ModelTypes.GRANITE_13B_CHAT_V2
    
    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.MAX_NEW_TOKENS: 100,
        GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
    }
    
    watsonx_Granite = WatsonxLLM(
    model_id=model_id.value,
    url=credentials.get("url"),
    apikey=credentials.get("apikey"),
    project_id=projectID,
    params=parameters
)

# Define the prompt
from langchain import hub
prompt = hub.pull("rlm/rag-prompt")



# Let the user upload a file via `st.file_uploader`.
uploaded_pdf = st.file_uploader(
    "Upload a document (.pdf)", type=("pdf")
)


    
# Ask the user for a question via `st.text_area`.
question = st.text_area(
    "Now ask a question about the document!",
    placeholder="Can you give me a short summary of this contract?",
    disabled=not uploaded_pdf,
)

from PyPDF2 import PdfReader
from langchain_ibm import WatsonxEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


if uploaded_pdf and question:
    st.write("Here")
    pdf_reader = PdfReader(uploaded_pdf)
    text = ""
    for page in pdf_reader.pages:
      text += page.extract_text()
    st.write("Splitting text")
    #Split the incoming text
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size= 500,
        chunk_overlap = 20, 
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    #Create embeddings
    st.write("Creating embeddings")
    embeddings = WatsonxEmbeddings(
    model_id="ibm/slate-30m-english-rtrvr",
    url=credentials["url"],
    apikey=credentials["apikey"],
    project_id=projectID
    )
    #Setup vectorstore using FAISS
    ids = [str(i) for i in range(0, len(chunks))]
    faissdb = FAISS.from_texts(chunks, embeddings, ids=ids)
    #Setup retreiver
    retriever = faissdb.as_retriever()
    st.write("Setup the chain")
    llm = LLM()
    rag_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
    response = rag_chain.invoke(question)
    st.text(response)


    