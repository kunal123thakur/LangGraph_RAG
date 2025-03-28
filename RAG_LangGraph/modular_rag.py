# app.py

import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import Graph

# -------------------- Load API Keys --------------------
load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")

# -------------------- LLM & Embeddings --------------------
llm = ChatGroq(api_key=GROK_API_KEY, model_name="Gemma2-9b-It")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# -------------------- Vector DB Setup --------------------
loader = DirectoryLoader("../data",glob="./*.txt",loader_cls=TextLoader)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
new_docs = text_splitter.split_documents(docs)

db = Chroma.from_documents(new_docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# -------------------- LangGraph Functions --------------------
def function_1(AgentState):
    message = AgentState["messages"]
    question = message[-1]

    complete_prompt = "Your task is to provide only the brief answer based on the user query. " \
                      "Don't include too much reasoning. Following is the user query: " + question

    response = llm.invoke(complete_prompt)
    AgentState['messages'].append(response.content)
    return AgentState

def function_2(AgentState):
    messages = AgentState['messages']
    question = messages[0]

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = retrieval_chain.invoke(question)
    return result

# -------------------- LangGraph Workflow --------------------
workflow = Graph()
workflow.add_node("LLM", function_1)
workflow.add_node("RAGtool", function_2)
workflow.add_edge('LLM', 'RAGtool')
workflow.set_entry_point("LLM")
workflow.set_finish_point("RAGtool")

app = workflow.compile()

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Corrective RAG LangGraph", page_icon="🤖")
st.title("🔍 Corrective RAG Pipeline with LangGraph")

user_input = st.text_input("Ask me anything related to your documents:")

if st.button("Run RAG Pipeline"):
    if user_input.strip() == "":
        st.warning("Please enter a question!")
    else:
        with st.spinner("Running LangGraph Workflow..."):
            inputs = {"messages": [user_input]}
            outputs = []

            for output in app.stream(inputs):
                for key, value in output.items():
                    outputs.append((key, value))

            # Display Outputs
            for key, value in outputs:
                st.subheader(f"📌 Output from {key}")
                st.code(value, language="text")
