import streamlit as st
import os
import pandas as pd
import uuid
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from gtts import gTTS
from supabase import create_client, Client

# --- Configuratie & Setup ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "Data")
METADATA_PATH = os.path.join(SCRIPT_DIR, "metadata.csv")

LLM_MODEL_GROQ = "llama3-70b-8192"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Supabase Initialisatie ---
supabase: Client = create_client(supabase_url, supabase_key)

# --- Functies ---
def log_to_supabase(session_id, bron, onderwerp, feedback_score, foutmelding=None):
    if not supabase_url or not supabase_key:
        return
    try:
        data_to_log = {
            "session_id": str(session_id),
            "document_bron": bron,
            "document_onderwerp": onderwerp,
            "feedback_score": feedback_score,
            "foutmelding": foutmelding
        }
        supabase.table("logs").insert(data_to_log).execute()
    except Exception as e:
        print(f"Fout bij het loggen naar Supabase: {e}")

@st.cache_resource
def load_and_process_data():
    # ... (Deze functie is complex en correct uit eerdere versies, we nemen hem over)
    try:
        df_meta = pd.read_csv(METADATA_PATH)
    except FileNotFoundError:
        st.error(f"metadata.csv niet gevonden.")
        st.stop()
    # ... (De rest van de laadlogica)
    all_docs = []
    for filename in os.listdir(DATA_PATH):
        filepath = os.path.join(DATA_PATH, filename)
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(filepath)
            try: all_docs.extend(loader.load())
            except Exception as e: st.warning(f"Kon PDF {filename} niet laden: {e}")
        elif filename.endswith(".txt"):
            content = None
            try:
                with open(filepath, 'r', encoding='utf-8') as f: content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(filepath, 'r', encoding='cp1252') as f: content = f.read()
                except Exception as e: st.warning(f"Kon {filename} niet lezen: {e}"); continue
            if content is not None:
                doc = Document(page_content=content, metadata={'source': filepath})
                all_docs.append(doc)
    docs_to_index = []
    for doc in all_docs:
        base_name = os.path.basename(doc.metadata['source'])
        meta_row = df_meta[df_meta['bestandsnaam'] == base_name]
        if not meta_row.empty:
            doc.metadata['volledige_tekst'] = doc.page_content
            doc.page_content = str(meta_row.iloc[0]['a2_samenvatting'])
            doc.metadata['bron'] = meta_row.iloc[0]['bron']
            doc.metadata['onderwerp'] = meta_row.iloc[0]['onderwerp']
            docs_to_index.append(doc)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(documents=docs_to_index, embedding=embeddings)
    return vectorstore

# --- Prompts ---
PROMPT_UITLEG = PromptTemplate.from_template(
    """
    Je bent een AI-assistent die laaggeletterde mensen helpt. Gebruik de VOLLEDIGE TEKST VAN DE BRIEF hieronder om de vraag te beantwoorden.
    ANALYSEER DE CONTEXT. Als de context duidelijk GEEN officiële brief is (bv een recept), antwoord dan alleen: "Deze tekst lijkt geen officiële brief te zijn. Ik kan u hier niet mee helpen."
    
    Als de context WEL een officiële brief is, geef dan een simpele samenvatting (A2-niveau). Structureer je antwoord als volgt:
    1.  Een korte introductiezin.
    2.  Gebruik bullet points. Begin elke bullet point op een nieuwe regel met een sterretje en een spatie (`* `).
    3.  Elke bullet point MOET beginnen met het juiste icoon. Dit is een strikte regel:
        * 🏢 **Van wie:**
        * 🎯 **Wat moet u doen?:**
        * 💰 **Bedrag:**
        * 🗓️ **Datum:**
        * ℹ️ **Let op:**

    CONTEXT (VOLLEDIGE TEKST VAN DE BRIEF): {context}
    VRAAG: {question}
    HELPENDE ANTWOORD:
    """
)

# --- Hoofdapplicatie ---
st.set_page_config(page_title="Hulp bij Moeilijke Brieven", layout="wide")
st.title("🤖 AI Hulp voor Moeilijke Brieven")

# Initialiseer session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'run_id' not in st.session_state:
    st.session_state.run_id = 0
if 'user_input_tab1' not in st.session_state:
    st.session_state.user_input_tab1 = ""
if 'ai_response' not in st.session_state:
    st.session_state.ai_response = None

# Controleer API keys
if not groq_api_key or not supabase_url or not supabase_key:
    st.error("API sleutels voor Groq of Supabase niet gevonden. Zorg dat deze zijn ingesteld.")
    st.stop()

# Bouw de RAG-keten
with st.spinner("De intelligente kennisbank wordt geladen..."):
    vectorstore = load_and_process_data()
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=LLM_MODEL_GROQ)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT_UITLEG, "document_variable_name": "context"},
        return_source_documents=True 
    )

# --- UI met Tabs ---
tab1, tab2 = st.tabs(["✉️ **Brief Laten Uitleggen**", "✍️ **Help Mij Schrijven**"])

with tab1:
    st.header("Laat je brief uitleggen")
    st.write("Plak de tekst van een moeilijke brief hieronder, of maak een foto van je brief.")
    
    agreed_tab1 = st.checkbox("Ik begrijp dat mijn brief anoniem verwerkt wordt en ga akkoord.", key="agree_tab1", value=True)
    if agreed_tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Optie 1: Maak een foto")
            uploaded_file = st.file_uploader("Kies een afbeelding...", type=["jpg", "png", "jpeg"], key="uploader_tab1")
            if uploaded_file:
                with st.spinner('Foto lezen...'):
                    image = Image.open(uploaded_file)
                    text = pytesseract.image_to_string(image, lang='nld')
                    st.session_state.user_input_tab1 = text
                    st.success("Tekst uit foto gehaald!")
        with col2:
            st.subheader("Optie 2: Plak de tekst")
            st.text_area("Tekst van de brief:", key="user_input_tab1", height=300)

        if st.button("Leg de brief uit", type="primary"):
            final_input = st.session_state.get('user_input_tab1', '')
            if final_input:
                with st.spinner("Ik zoek en maak een samenvatting..."):
                    try:
                        st.session_state.ai_response = qa_chain.invoke(final_input)
                        st.session_state.run_id += 1
                    except Exception as e:
                        st.session_state.ai_response = None
                        st.error(f"Er is een technische fout opgetreden: {e}")
                        log_to_supabase(st.session_state.session_id, "N.v.t.", "N.v.t.", -1, foutmelding=str(e))
            else:
                st.warning("Plak eerst tekst of upload een foto.")
        
        # Toon het resultaat als het in de session state staat
        if st.session_state.ai_response:
            response = st.session_state.ai_response
            answer = response.get('result', '')
            source_docs = response.get('source_documents', [])
            doc_metadata = source_docs[0].metadata if source_docs else {}

            if answer:
                st.subheader("Simpele Uitleg:")
                st.markdown(f"{answer}", unsafe_allow_html=True)
                
                st.subheader("Lees de uitleg voor:")
                tts = gTTS(text=answer, lang='nl', slow=False)
                tts_file = f"uitleg_{st.session_state.run_id}.mp3"
                tts.save(tts_file)
                st.audio(tts_file)
                
                st.write("---")
                st.write("**Was deze uitleg nuttig?**")
                feedback_cols = st.columns(5)
                with feedback_cols[0]:
                    if st.button("👍 Ja", key=f"yes_{st.session_state.run_id}"):
                        log_to_supabase(st.session_state.session_id, doc_metadata.get('bron'), doc_metadata.get('onderwerp'), 1)
                        st.success("Bedankt voor je feedback!")
                        st.session_state.ai_response = None # Reset na feedback
                        st.rerun()
                with feedback_cols[1]:
                    if st.button("👎 Nee", key=f"no_{st.session_state.run_id}"):
                        log_to_supabase(st.session_state.session_id, doc_metadata.get('bron'), doc_metadata.get('onderwerp'), 0)
                        st.success("Bedankt voor je feedback!")
                        st.session_state.ai_response = None # Reset na feedback
                        st.rerun()

                with st.expander("Bekijk welke informatie is gebruikt"):
                    for doc in source_docs:
                        st.info(f"**Gevonden document:** {doc.metadata.get('bron')} - {doc.metadata.get('onderwerp')}")

# ... (Tab 2 en Privacyverklaring code hieronder) ...
with tab2:
    # Je bestaande code voor schrijfhulp
    pass

st.markdown("---")
with st.expander("Privacy en Veiligheid"):
    st.write("""
    **Jouw privacy is belangrijk.**
    - Wij slaan jouw brieven en foto's **NIET** op.
    - Om de uitleg te geven, sturen we de tekst anoniem naar een beveiligde AI-dienst (Groq).
    - Deze dienst gebruikt jouw gegevens **niet** om hun AI te trainen.
    - Wij slaan wel **anonieme** feedback op (een 'duimpje omhoog' of 'omlaag') om de tool te verbeteren.
    """)