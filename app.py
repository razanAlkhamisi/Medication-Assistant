import os
import requests
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI as LangChainOpenAI
from langchain.agents.agent_types import AgentType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
import easyocr
import time
from PIL import Image
from requests.adapters import HTTPAdapter, Retry
from gtts import gTTS
import pygame
from PyPDF2 import PdfReader
import streamlit as st
from io import BytesIO
from openai import OpenAI


# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize pygame mixer
pygame.mixer.init()

# Track last read message
if 'last_read_message' not in st.session_state:
    st.session_state.last_read_message = ""

# Add a flag to indicate if we should read a message after rerun
if 'read_after_rerun' not in st.session_state:
    st.session_state.read_after_rerun = False
    st.session_state.message_to_read = ""

# Add a flag to track if voice chat is active
if 'voice_chat_active' not in st.session_state:
    st.session_state.voice_chat_active = False

def speak_text_gtts(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            f.write(fp.read())
            temp_audio_path = f.name
        
        # Load and play the audio
        pygame.mixer.music.load(temp_audio_path)
        pygame.mixer.music.play()
        
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        return temp_audio_path
    except Exception as e:
        st.error(f"Voice playback error: {e}")
        return None

# Page config to make the app look nicer
st.set_page_config(
    page_title="Medicine Assistant",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS to style the app like Telegram
st.markdown("""
<style>
/* Main application background with gradient */
.stApp {
    background: linear-gradient(135deg, #8ab5e8, #d1e1fb);
    display: flex;
    flex-direction: column;
    height: 100vh;
    font-family: 'Poppins', sans-serif;
}

/* Streamlit header styling */
.stApp header {
    background: linear-gradient(90deg, #3a6ea5, #5682a3, #3a6ea5) !important;
    color: white !important;
    padding: 12px 24px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15) !important;
    position: fixed !important;
    top: 0 !important;
    width: 100% !important;
    z-index: 1000 !important;
    backdrop-filter: blur(8px) !important;
}

/* Adjust top spacing to prevent content overlap */
.stApp > div:first-child {
    padding-top: 90px !important;
    padding-bottom: 80px !important;
}

/* Fixed bottom input bar with glass morphism */
[data-testid="stHorizontalBlock"].footer {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    background: rgba(86, 130, 163, 0.85) !important;
    backdrop-filter: blur(8px) !important;
    padding: 12px 18px !important;
    z-index: 1000 !important;
    border-top: 1px solid rgba(255, 255, 255, 0.2) !important;
    box-shadow: 0 -5px 20px rgba(0, 0, 0, 0.08) !important;
}

/* ===== MAIN CONTENT AREA STYLING ===== */
/* Main content container - makes sure content is prominent and readable */
.main .block-container {
    max-width: 1100px !important;
    padding: 2rem 3rem !important;
    margin: 0 auto !important;
    background: rgba(255, 255, 255, 0.92) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1) !important;
}

/* Make text in main area more readable */
.main p, .main li {
    font-size: 16px !important;
    line-height: 1.6 !important;
    color: #37474f !important;
}

/* Headings in main content area */
.main h1 {
    font-size: 28px !important;
    font-weight: 600 !important;
    color: #2c5282 !important;
    margin-bottom: 1.5rem !important;
    border-bottom: 2px solid #bbdefb !important;
    padding-bottom: 0.5rem !important;
}

.main h2 {
    font-size: 22px !important;
    font-weight: 500 !important;
    color: #3a6ea5 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 1rem !important;
}

.main h3 {
    font-size: 18px !important;
    font-weight: 500 !important;
    color: #4b7bab !important;
    margin-top: 1rem !important;
}



/* ===== SIDEBAR STYLING ===== */
/* Sidebar container with glass morphism effect */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(10px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.5) !important;
}

/* Sidebar header with gradient */
[data-testid="stSidebar"] .sidebar-content {
    background: linear-gradient(180deg, rgba(58, 110, 165, 0.1) 0%, rgba(255, 255, 255, 0) 100%) !important;
}

/* Sidebar title styling */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2 {
    color: #2c5282 !important;
    border-bottom: 1px solid rgba(58, 110, 165, 0.3) !important;
    padding-bottom: 10px !important;
    margin-bottom: 15px !important;
}

/* Sidebar section dividers */
[data-testid="stSidebar"] hr {
    margin: 20px 0 !important;
    border-color: rgba(58, 110, 165, 0.2) !important;
}

/* Sidebar menu items */
[data-testid="stSidebar"] .sidebar-content > div {
    padding: 15px !important;
}

/* Interactive elements in sidebar */
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stMultiselect {
    margin-bottom: 12px !important;
    width: 100% !important;
}

/* Improve sidebar widget styling */
[data-testid="stSidebar"] .stRadio > div,
[data-testid="stSidebar"] .stCheckbox > div {
    background: rgba(255, 255, 255, 0.5) !important;
    padding: 10px !important;
    border-radius: 6px !important;
    margin-bottom: 10px !important;
    border: 1px solid rgba(58, 110, 165, 0.2) !important;
}

/* Sidebar toggle button styling */
button[kind="header"] {
    background: linear-gradient(135deg, #4b7bab, #3a6ea5) !important;
    color: #2c5282 !important;
    border: none !important;
    border-radius: 4px !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] h6 {
    color: #2c5282 !important; 
    /* color: #1a365d !important; */
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: inherit !important; 
}
                       
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stTooltip {
    color: #4a6fa5 !important; 
}

[data-testid="stSidebar"] h1 + p,
[data-testid="stSidebar"] h2 + p,
[data-testid="stSidebar"] h3 + p {
    color: #3a5f8d !important; 
    font-size: 0.9rem !important;
    margin-top: -10px !important; 
}
            
/* Input field styling */
/* All text inputs */
.stTextInput input, 
.stNumberInput input,
.stDateInput input,
.stTimeInput input,
textarea.st-bq,
.stSelectbox select,
.stMultiselect select {
    border: none !important;
    border-radius: 4px !important; /* Less rounded corners */
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1) !important;
    padding-left: 12px !important;
    transition: all 0.3s ease !important;
    background: #ffffff !important;
    color: #37474f !important; /* Match the chat text color */
}

.stTextInput input {
    height: 42px !important;
}

/* Target specific text input by ID as well */
#text_input_1 input {
    background: #ffffff !important;
    color: #37474f !important;
    border-radius: 4px !important;
}

.stTextInput input:focus {
    box-shadow: 0 4px 15px rgba(59, 89, 152, 0.25) !important;
    transform: translateY(-2px) !important;
}

/* Send button styling */
.stButton button {
    background: linear-gradient(135deg, #4b7bab, #3a6ea5) !important;
    color: white !important;
    font-weight: 600 !important;
    height: 42px !important;
    border-radius: 24px !important;
    padding: 0 24px !important;
    border: none !important;
    box-shadow: 0 4px 10px rgba(59, 89, 152, 0.2) !important;
    transition: all 0.3s ease !important;
}

.stButton button:hover {
    background: linear-gradient(135deg, #3a6ea5, #2c5282) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 15px rgba(59, 89, 152, 0.3) !important;
}

/* Compact input bar with centered positioning */
[data-testid="stHorizontalBlock"] {
    position: fixed;
    bottom: 15px;
    left: 50%;
    transform: translateX(-50%);
    width: 90%;
    max-width: 800px;
    padding: 10px 15px !important;
    border-radius: 28px;
    box-shadow: 0 5px 25px rgba(0, 0, 0, 0.15);
    z-index: 1000;
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.3);
}

/* Main container with elegant design */
.main-container {
    display: flex;
    flex-direction: column;
    height: 100%;
    max-width: 850px;
    margin: 0 auto;
    background: rgba(255, 255, 255, 0.8);
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.2);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.18);
    position: relative;
    overflow: hidden;
}

/* Header with gradient and subtle animation */
.app-header {
    background: linear-gradient(90deg, #3a6ea5, #5682a3, #3a6ea5);
    background-size: 200% 100%;
    color: white;
    padding: 16px 24px;
    border-radius: 16px 16px 0 0;
    display: flex;
    align-items: center;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    animation: gradientAnimation 8s ease infinite;
}

@keyframes gradientAnimation {
    0% {background-position: 0% 50%}
    50% {background-position: 100% 50%}
    100% {background-position: 0% 50%}
}

.app-title {
    margin: 0;
    font-size: 1.6rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* Chat container with subtle pattern */
.chat-container {
    padding: 24px;
    overflow-y: auto;
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    max-height: calc(100vh - 220px);
    background-image: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%235682a3' fill-opacity='0.05' fill-rule='evenodd'/%3E%3C/svg%3E");
}

/* Enhanced Cards for Important Information */
.info-card {
    background: white;
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    border-left: 4px solid #3a6ea5;
}

.info-card h3 {
    margin-top: 0;
    color: #3a6ea5;
}

.info-card.warning {
    border-left: 4px solid #ff9800;
}

.info-card.success {
    border-left: 4px solid #4CAF50;
}

.info-card.error {
    border-left: 4px solid #f44336;
}

/* Beautiful message bubbles with animations */
.user-message {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    border-radius: 18px 18px 0 18px;
    padding: 14px 18px;
    margin: 8px 0;
    max-width: 80%;
    align-self: flex-end;
    float: right;
    clear: both;
    color: #37474f;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.5);
    animation: fadeInRight 0.3s ease;
    position: relative;
}

.user-message:after {
    content: "";
    position: absolute;
    bottom: 0;
    right: 0;
    width: 12px;
    height: 12px;
    background: linear-gradient(135deg, transparent 50%, #bbdefb 50%);
    border-bottom-right-radius: 14px;
}

.bot-message {
    background: linear-gradient(135deg, #ffffff, #f5f5f5);
    border-radius: 18px 18px 18px 0;
    padding: 14px 18px;
    margin: 8px 0;
    max-width: 80%;
    align-self: flex-start;
    float: left;
    clear: both;
    color: #37474f;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.5);
    animation: fadeInLeft 0.3s ease;
    position: relative;
}

.bot-message:after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    width: 12px;
    height: 12px;
    background: linear-gradient(135deg, #f5f5f5 50%, transparent 50%);
    border-bottom-left-radius: 14px;
}

@keyframes fadeInRight {
    from {
        opacity: 0;
        transform: translateX(20px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes fadeInLeft {
    from {
        opacity: 0;
        transform: translateX(-20px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

/* Message text styling */
.message-text {
    margin: 0;
    line-height: 1.5;
    font-size: 15px;
}

/* Enhanced Tables in Main Content */
.stDataFrame {
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
    border: 1px solid #e0e0e0 !important;
}

.stDataFrame th {
    background: linear-gradient(90deg, #4b7bab, #5682a3) !important;
    color: white !important;
    font-weight: 500 !important;
    padding: 12px 15px !important;
}

.stDataFrame td {
    padding: 10px 15px !important;
    border-bottom: 1px solid #f0f0f0 !important;
}

.stDataFrame tr:nth-child(even) {
    background-color: #f9f9f9 !important;
}

.stDataFrame tr:hover {
    background-color: #f0f7ff !important;
}

/* Add visual cues for navigation */
.main .stMarkdown a {
    color: #3a6ea5 !important;
    text-decoration: none !important;
    border-bottom: 1px dotted #3a6ea5 !important;
    transition: all 0.2s !important;
}

.main .stMarkdown a:hover {
    color: #2c5282 !important;
    border-bottom: 1px solid #2c5282 !important;
}

/* Button hover effects */
div.stButton > button {
    border-radius: 22px;
    background: linear-gradient(135deg, #4b7bab, #3a6ea5);
    color: white;
    transition: all 0.3s;
    border: none;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 12px rgba(59, 89, 152, 0.25);
}

div.stButton {
    margin-top: 28px;
}

/* File uploader styling */
.stFileUploader > div > button {
    border-radius: 22px;
    background: linear-gradient(135deg, #4b7bab, #3a6ea5);
    color: white;
    transition: all 0.3s;
    border: none;
}

.stFileUploader > div > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 12px rgba(59, 89, 152, 0.25);
}

/* Loading animation and style improvements */
.stProgress > div > div {
    background-color: #5682a3 !important;
    height: 4px !important;
    border-radius: 2px !important;
}

/* Improved expander style */
.stExpander {
    border: none;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    border-radius: 12px;
    overflow: hidden;
}

.stExpander > div:first-child {
    background: linear-gradient(90deg, #f9f9f9, #ffffff);
    border-radius: 8px;
}

/* Time indicators for messages */
.message-time {
    font-size: 11px;
    color: #9e9e9e;
    margin-top: 5px;
    text-align: right;
}

/* Typing indicator animation */
.typing-indicator {
    display: flex;
    align-items: center;
    margin: 10px 0;
}

.typing-dot {
    width: 8px;
    height: 8px;
    margin: 0 2px;
    background-color: #5682a3;
    border-radius: 50%;
    opacity: 0.6;
    animation: typingAnimation 1.4s infinite ease-in-out;
}

.typing-dot:nth-child(1) {
    animation-delay: 0s;
}

.typing-dot:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-dot:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes typingAnimation {
    0%, 60%, 100% {
        transform: translateY(0);
    }
    30% {
        transform: translateY(-5px);
    }
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb {
    background: rgba(86, 130, 163, 0.5);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(86, 130, 163, 0.8);
}

/* Clear floats after messages */
.clearfix::after {
    content: "";
    clear: both;
    display: table;
}

/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');
</style>
""", unsafe_allow_html=True)
# Functions from the original code
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def preprocess_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

def setup_vector_db(texts):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store

def build_qa_chain_from_pdf(pdf_file):
    text = extract_text_from_pdf(pdf_file)
    if not text.strip():
        raise ValueError("Empty or unreadable PDF text.")
    chunks = preprocess_text(text)
    vector_store = setup_vector_db(chunks)
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever), vector_store

# OpenFDA Setup - Updated version from second code
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

def search_openfda(query: str) -> str:
    """
    Searches the OpenFDA API for drug information and returns a summarized, safe-length result.
    """

    import requests

    try:
        # Make API request to OpenFDA
        url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{query}&limit=1"
        response = requests.get(url)
        data = response.json()

        if "results" not in data or not data["results"]:
            return f"No FDA data found for '{query}'."

        result = data["results"][0]

        # Extract only relevant fields
        sections = {
            "Brand Name": ", ".join(result.get("openfda", {}).get("brand_name", [])),
            "Manufacturer": ", ".join(result.get("openfda", {}).get("manufacturer_name", [])),
            "Purpose": result.get("purpose", ["N/A"])[0],
            "Indications and Usage": result.get("indications_and_usage", ["N/A"])[0][:500],  # limit section size
            "Warnings": result.get("warnings", ["N/A"])[0][:500],
            "Dosage and Administration": result.get("dosage_and_administration", ["N/A"])[0][:500]
        }

        # Format response
        summary = "\n".join([f"**{key}:** {value}" for key, value in sections.items() if value])
        return summary

    except Exception as e:
        return f"An error occurred while accessing OpenFDA: {str(e)}"
    
# EasyOCR for image text extraction
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)

def extract_drug_name(image):
    reader = load_ocr_reader()
    results = reader.readtext(np.array(image))
    if not results:
        return None
    texts_with_heights = [(text, abs(bbox[3][1] - bbox[0][1])) for bbox, text, conf in results]
    return max(texts_with_heights, key=lambda x: x[1])[0].lower().strip()

# GPT Query Type Detection - Updated version from second code
def detect_query_type_with_gpt(query):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a query classifier. Classify the user's query strictly as one of the following:\n"
                        "'medicine' – if it's asking about a drug name, dosage, effects, etc.\n"
                        "'symptom' – if it's asking about medical symptoms or conditions.\n"
                        "'greeting' – if it's a polite message like hello, hi, thank you, bye, etc.\n"
                        "'other' – for anything else.\n"
                        "Only respond with one word: 'medicine', 'symptom', 'greeting', or 'other'."
                    )
                },
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        st.error(f"Query type detection failed: {e}")
        return "other"

# Small talk tool from second code
def small_talk_tool_func(query: str) -> str:
    prompt = f"""You are a friendly AI medical assistant. This is a casual, non-medical message. 
Respond in a warm and conversational tone.

User: {query}
Assistant:"""
    llm = LangChainOpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
    return llm(prompt)

# Modified Audio Input with Whisper for continuous voice chat
def recognize_speech_openai():
    fs = 16000
    seconds = 5
    
    while st.session_state.voice_chat_active:
        status_placeholder = st.empty()
        status_placeholder.info("I’m listening to you. Whenever you're ready to stop, just say bye.")
        
        try:
            myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
            sd.wait()
            status_placeholder.info("Hang tight! I'm getting your answer ready.")
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wavfile.write(f.name, fs, (myrecording * 32767).astype(np.int16))
                with open(f.name, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
            
            user_input = transcript.text
            status_placeholder.empty()
            
            # Check conversation ending words
            if any(word in user_input.lower() for word in ["باى", "bye"]):
                st.session_state.voice_chat_active = False
                farewell_msg = "Goodbye! Stay safe and take care!."
                st.session_state.messages.append({"role": "assistant", "content": farewell_msg})
                st.session_state.read_after_rerun = True
                st.session_state.message_to_read = farewell_msg
                st.rerun()
                break
            
            # If not a termination word, continue processing.
            if user_input.strip():
                st.session_state.messages.append({"role": "user", "content": user_input})
                response = process_query(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                st.session_state.read_after_rerun = True
                st.session_state.message_to_read = response
                st.rerun()
                
        except Exception as e:
            status_placeholder.error(f"An error occurred in the audio recording: {e}")
            time.sleep(2)
            status_placeholder.empty()
            break

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Initialize LangChain agent - Updated with small talk tool
openfda_tool = Tool(
    name="search_openfda",
    func=search_openfda,
    description="Use this to get official medicine information like usage, dosage, warnings, or manufacturer. Always use for any medicine-related query."
)

small_talk_tool = Tool(
    name="small_talk_handler",
    func=small_talk_tool_func,
    description="Use this to respond to greetings, thanks, or general small talk not related to medicine or symptoms don't ever use it with medicine information."
)

tools = [openfda_tool, small_talk_tool]
llm = LangChainOpenAI(
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_kwargs={"max_tokens": 256}
)
memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Function to process queries - Updated to handle greetings
def process_query(query):
    query_type = detect_query_type_with_gpt(query)
    
    if query_type == "greeting":
        return small_talk_tool_func(query)
    elif query_type == "symptom" or query_type == "medicine":
        try:
            return agent.run(query)
        except Exception as e:
            return f"Agent error: {str(e)}"
    elif st.session_state.qa_chain:
        try:
            return st.session_state.qa_chain.run(query)
        except Exception as e:
            return f"PDF QA error: {str(e)}"
    return "Sorry, I can only help with medicine questions."

# App Layout
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# App header
st.markdown("""
<div class="app-header">
    <h1 class="app-title">💊 Medicine Assistant</h1>
</div>
""", unsafe_allow_html=True)

# Chat area
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message"><p class="message-text">{message["content"]}</p></div><div class="clearfix"></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message"><p class="message-text">{message["content"]}</p></div><div class="clearfix"></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Check if we need to read a message after rerun
if st.session_state.read_after_rerun:
    speak_text_gtts(st.session_state.message_to_read)
    st.session_state.read_after_rerun = False
    st.session_state.last_read_message = st.session_state.message_to_read

# Sidebar for file uploads and voice input
with st.sidebar:
    st.title("Tools")
    
    # Image upload
    st.subheader("Upload Medicine Image")
    uploaded_image = st.file_uploader("Upload an image of your medicine", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", width=300)
        
        if st.button("Extract Medicine Name"):
            with st.spinner("Analyzing image..."):
                drug_name = extract_drug_name(image)
                if drug_name:
                    # Add messages to chat
                    st.session_state.messages.append({"role": "user", "content": f"I have a medicine called {drug_name}. What can you tell me about it?"})
                    
                    # Get response
                    response = process_query(drug_name)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Set flag to read after rerun
                    st.session_state.read_after_rerun = True
                    st.session_state.message_to_read = response
                    
                    # Force a rerun to update the chat UI
                    st.rerun()
                else:
                    st.error("Couldn't detect a medicine name from the image. Please try uploading a clearer one.")

    # PDF upload
    st.subheader("Upload Medical Document")
    uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_pdf is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                try:
                    qa_chain, _ = build_qa_chain_from_pdf(uploaded_pdf)
                    st.session_state.qa_chain = qa_chain
                    
                    # Add confirmation message to chat
                    response = "Great, PDF loaded successfully. You can now ask questions about its content."
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Set flag to read after rerun
                    st.session_state.read_after_rerun = True
                    st.session_state.message_to_read = response
                    
                    # Force a rerun to update the chat UI
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
    
    # Voice input
    st.subheader("Voice Chat")
    if not st.session_state.voice_chat_active:
        if st.button("Start Voice Chat"):
            st.session_state.voice_chat_active = True
            st.session_state.messages.append({"role": "assistant", "content": "Voice chat started. You can speak now. Say 'bye' to end the voice chat."})
            st.rerun()
    else:
        if st.button("Stop Voice Chat"):
            st.session_state.voice_chat_active = False
            st.session_state.messages.append({"role": "assistant", "content": "Voice chat ended."})
            st.rerun()
    
    # This will start the recognition when the flag is set
    if st.session_state.voice_chat_active:
        recognize_speech_openai()

# Input area at the bottom
with st.container():
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input("Type your message...", key="user_message_input")
    
    with col2:
        send_button = st.button("Send")
    
    if send_button and user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Process the query
        response = process_query(user_input)
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Set flag to read after rerun
        st.session_state.read_after_rerun = True
        st.session_state.message_to_read = response
        
        # Force a rerun to update the chat UI
        st.rerun()

# Welcome message on first load
if len(st.session_state.messages) == 0:
    welcome_msg = "Hi there! Welcome to the Medicine Assistant. How can I help you?"
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    
    # Set flag to read after rerun
    st.session_state.read_after_rerun = True
    st.session_state.message_to_read = welcome_msg
    
    # Automatically start voice chat after welcome
    st.session_state.voice_chat_active = True
    
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)