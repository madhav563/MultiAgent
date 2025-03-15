import streamlit as st
import tempfile
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema import Document  # Add this import

st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="🔍")
st.title("Multi-Agent RAG Research Assistant")
st.subheader("Upload files or enter URLs for advanced research queries")

with st.sidebar:
    grok_api_key = st.text_input("Enter your Grok API Key", type="password")

# Inputs
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
youtube_url = st.text_input("YouTube URL (optional)")
web_url = st.text_input("Website URL (optional)")
query = st.text_input("Enter your research question")
num_docs = st.number_input("Number of documents to retrieve", min_value=1, max_value=10, value=3)

# Agent prompts
extractor_prompt = PromptTemplate(
    input_variables=["text"],
    template="Extract key insights from this content: {text}"
)
rag_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="Using this context: {context}, answer: {query}"
)
critic_prompt = PromptTemplate(
    input_variables=["answer"],
    template="Critique this answer for accuracy and coherence, suggest improvements: {answer}"
)

if st.button("Process and Answer"):
    if not grok_api_key.strip() or not query.strip():
        st.error("Please provide a Grok API Key and query")
    else:
        try:
            with st.spinner("Processing sources..."):
                # Initialize LLM
                llm = ChatGroq(model="Gemma2-9b-it", groq_api_key=grok_api_key)
                
                # Agent 1: Extractor - Process multiple sources
                documents = []
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

                # PDFs
                for file in uploaded_files or []:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(file.read())
                        tmp_file_path = tmp_file.name
                    loader = PyPDFLoader(tmp_file_path)
                    documents.extend(loader.load_and_split(text_splitter))

                # YouTube
                if youtube_url and "youtube.com" in youtube_url:
                    video_id = youtube_url.split("v=")[1].split("&")[0]
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    transcript_text = " ".join([entry["text"] for entry in transcript])
                    documents.append(Document(page_content=transcript_text))

                # Website
                if web_url:
                    loader = UnstructuredURLLoader(urls=[web_url], ssl_verify=False)
                    documents.extend(loader.load_and_split(text_splitter))

                # Extract insights
                extractor_chain = LLMChain(llm=llm, prompt=extractor_prompt)
                extracted_texts = [extractor_chain.run(doc.page_content) for doc in documents]

                # Agent 2: RAG - Build vector store and retrieve context
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_store = FAISS.from_texts(extracted_texts, embeddings)
                retriever = vector_store.as_retriever(search_kwargs={"k": num_docs})
                context = " ".join([doc.page_content for doc in retriever.get_relevant_documents(query)])

                # Generate initial answer
                rag_chain = LLMChain(llm=llm, prompt=rag_prompt)
                initial_answer = rag_chain.run(context=context, query=query)

                # Agent 3: Critic - Refine the answer
                critic_chain = LLMChain(llm=llm, prompt=critic_prompt)
                final_answer = critic_chain.run(answer=initial_answer)

                st.success("Research Complete!")
                st.write("Final Answer:", final_answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")
