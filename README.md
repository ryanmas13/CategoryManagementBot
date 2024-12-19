# 📄 Document question answering template

A simple Streamlit app that answers questions about an uploaded document via OpenAI's GPT-4o-mini.

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

Additional information about Llamaindex chat engines: 
LlamaIndex has four different chat engines:

1. Condense question engine: Always queries the knowledge base. Can have trouble with meta questions like “What did I previously ask you?”

2. Context chat engin": Always queries the knowledge base and uses retrieved text from the knowledge base as context for following queries. The retrieved context from previous queries can take up much of the available context for the current query.

3. ReAct agent: Chooses whether to query the knowledge base or not. Its performance is more dependent on the quality of the LLM. You may need to coerce the chat engine to correctly choose whether to query the knowledge base.

4. OpenAI agent: Chooses whether to query the knowledge base or not—similar to ReAct agent mode, but uses OpenAI’s built-in fuOpenAI'salling capabilities.

We will use the OpenAI agent here. 