import streamlit as st
import time
import base64
from pathlib import Path
import nltk
from langchain_openai import OpenAIEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone as PineconeClient
from langchain_community.retrievers import PineconeHybridSearchRetriever
import os
from dotenv import load_dotenv
from functools import lru_cache
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
import uuid
import asyncio

load_dotenv()

# Lazy download nltk data only if not already downloaded
if "nltk_data" not in st.session_state:
    nltk.download('punkt')
    st.session_state.nltk_data = True

# Function to convert an image file to a base64 string
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Paths to the avatar images on your computer (Replace with your actual paths)
user_avatar_path = Path("../avatar/human.png")
assistant_avatar_path = Path("../avatar/ai.png")

# Convert images to base64
user_avatar_base64 = image_to_base64(user_avatar_path)
assistant_avatar_base64 = image_to_base64(assistant_avatar_path)

# CSS for fixed-height scrollable chat area and alignment
st.markdown("""
    <style>
    .chat-container {
        height: 400px;
        overflow-y: auto;
    }
    .user-message {
        display: flex;
        justify-content: flex-end;
        align-items: center;
    }
    .assistant-message {
        display: flex;
        justify-content: flex-start;
        align-items: center;
    }
    .message-bubble {
        padding: 10px;
        border-radius: 15px;
        margin: 5px;
        max-width: 70%;
    }
    .user-bubble {
        background-color: #DCF8C6;
        color: black;
    }
    .assistant-bubble {
        background-color: #ADE8F4;
        color: black;
    }
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Cache heavy operations like loading embeddings
def load_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-large")

def bm25_encoder():
    return BM25Encoder().default()

def initialize_pinecone_client(api_key, index_name):
    pc = PineconeClient(api_key=api_key)
    index = pc.Index(index_name)
    return index

def initialize_llm(callback_manager):
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        streaming=True,
        callback_manager=callback_manager,
        request_timeout=30,
        )
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

if "embeddings" not in st.session_state:
    st.session_state.embeddings = load_embeddings()
embeddings = st.session_state.embeddings

if "pinecone_index" not in st.session_state:
    st.session_state.pinecone_index = initialize_pinecone_client(os.getenv('PINECONE_API_KEY'), os.getenv('PINECONE_INDEX_NAME'))
index = st.session_state.pinecone_index

if "bm25_encoder" not in st.session_state:
    st.session_state.bm25_encoder = bm25_encoder()
bm25_encoder = st.session_state.bm25_encoder

if "llm" not in st.session_state:
    st.session_state.llm = initialize_llm(callback_manager)
llm = st.session_state.llm

retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)
# Create a cached version of the retriever
@lru_cache(maxsize=1000)
def cached_retrieval(query: str):
    return retriever.get_relevant_documents(query)


# Optimize prompts by reducing tokens
CONTEXTUALIZE_SYSTEM_PROMPT = """Reformulate the latest question to be standalone, \
considering chat history context. Return as is if already standalone."""

QA_SYSTEM_PROMPT = """You are BEA, a bank SME loan specialist. \
Answer using provided context only. Say "I don't know" if unsure. \
Answer in a professional and detailed manner. \
Prefix sensitive info with "According to BPI". Capitalize abbreviations.

{context}"""

# Create optimized prompt templates
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])


# Optimized history-aware retriever
def create_optimized_retriever(query: str, chat_history: Optional[list] = None) -> list:
    """Optimized retriever that uses caching and minimal token usage"""
    if not chat_history:
        return cached_retrieval(query)

    # Only contextualize if there's relevant chat history
    contextualized_q = llm.invoke(
        contextualize_q_prompt.format(
            chat_history=chat_history[-3:],  # Only use last 3 messages
            input=query
        )
    )
    return cached_retrieval(contextualized_q.content)


# Optimized question-answer chain
question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt,
    document_variable_name="context",
)


# Session management with TTL cache
class SessionManager:
    def __init__(self, ttl_seconds: int = 3600):
        self.sessions = {}
        self.ttl = ttl_seconds

    def get_session(self, session_id: Optional[str] = None) -> tuple:
        if session_id is None:
            session_id = str(uuid.uuid4())
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMessageHistory()
        return session_id, self.sessions[session_id]


session_mgr = SessionManager()


async def process_query(query: str, session_id: Optional[str] = None) -> str:
    """Main async function to process queries"""
    session_id, history = session_mgr.get_session(session_id)

    # Get relevant documents
    docs = create_optimized_retriever(query, history.messages)

    # Generate response
    response = question_answer_chain.invoke({
        "input": query,
        "chat_history": history.messages[-3:],  # Only use recent history
        "context": docs
    })

    # Update history
    history.add_user_message(query)
    history.add_ai_message(response)

    return response

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []


# Function to add messages to the chat history
def add_message(role, message):
    st.session_state['chat_history'].append({"role": role, "content": message})


# Display chat history
chat_container = st.empty()


def display_chat():
    with chat_container.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)  # Start the scrollable area
        for entry in st.session_state['chat_history']:  # Display messages in chronological order
            role = entry["role"]
            content = entry["content"]

            if role == "user":
                st.markdown(
                    f"""
                    <div class="user-message">
                        <div class="message-bubble user-bubble">{content}</div>
                        <img src="data:image/png;base64,{user_avatar_base64}" class="avatar" alt="User Avatar">
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="assistant-message">
                        <img src="data:image/png;base64,{assistant_avatar_base64}" class="avatar" alt="Assistant Avatar">
                        <div class="message-bubble assistant-bubble">{content}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)  # End the scrollable area


# Display chat initially
display_chat()

# User input with st.chat_input
if user_input := st.chat_input("Type your message here..."):
    # Add user's message
    add_message("user", user_input)

    # Re-display chat with the new message
    display_chat()

    # Simulate assistant response with a delay
    with st.spinner("Assistant is typing..."):
        assistant_reply = asyncio.run(process_query(user_input))
        add_message("assistant", assistant_reply)

    # Re-display chat with the assistant's response
    display_chat()