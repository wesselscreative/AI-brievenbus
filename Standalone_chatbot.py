import streamlit as st
import os
import pandas as pd
import uuid # Voor unieke sessie IDs
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyMuPDFLoader
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
load_dotenv() # Laadt omgevingsvariabelen uit het .env bestand

# API sleutels voor Groq en Supabase
groq_api_key = os.getenv("GROQ_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

# Padconfiguratie
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "Data")
METADATA_PATH = os.path.join(SCRIPT_DIR, "metadata.csv")

# LLM en Embedding Modelnamen
LLM_MODEL_GROQ = "llama3-70b-8192" # Gebruikt het krachtige Llama 3 model van Groq
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Voor efficiënte embeddings

# --- Supabase Initialisatie ---
# Controleer of de Supabase keys zijn ingesteld; anders stop de app.
if supabase_url and supabase_key:
    supabase: Client = create_client(supabase_url, supabase_key)
else:
    supabase = None
    st.warning("Supabase API sleutels niet gevonden. Logging zal niet werken.")


# --- Functies ---

def log_to_supabase(session_id, bron, onderwerp, feedback_score, tts_gebruikt=False, foutmelding=None):
    """Slaat een anonieme log op in de Supabase 'logs' tabel."""
    if not supabase: # Als de Supabase client niet is geïnitialiseerd, doe niets.
        return
    try:
        data_to_log = {
            "session_id": str(session_id), 
            "document_bron": bron, 
            "document_onderwerp": onderwerp,
            "feedback_score": feedback_score, 
            "tts_gebruikt": tts_gebruikt, 
            "foutmelding": foutmelding
        }
        supabase.table("logs").insert(data_to_log).execute()
    except Exception as e:
        # Fouten bij het loggen worden alleen in de console geprint, zodat de app blijft werken.
        print(f"Fout bij het loggen naar Supabase: {e}")

@st.cache_resource
def load_and_process_data():
    """
    Laadt alle documenten (TXT & PDF), verrijkt ze met metadata, en indexeert de A2-samenvattingen.
    Deze functie is robuust tegen encodingfouten.
    """
    try:
        df_meta = pd.read_csv(METADATA_PATH)
    except FileNotFoundError:
        st.error(f"metadata.csv niet gevonden. Zorg dat dit bestand in je projectmap staat.")
        st.stop() # Stop de app als de metadata-file mist

    all_docs = []
    for filename in os.listdir(DATA_PATH):
        filepath = os.path.join(DATA_PATH, filename)
        
        # Bestandstypen verwerken (PDF en TXT)
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(filepath)
            try:
                all_docs.extend(loader.load())
            except Exception as e:
                st.warning(f"Kon PDF {filename} niet laden: {e}")
        elif filename.endswith(".txt"):
            content = None
            try:
                # Probeer UTF-8 encoding
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    # Fallback naar CP1252 encoding voor Windows-tekstbestanden
                    with open(filepath, 'r', encoding='cp1252') as f:
                        content = f.read()
                except Exception as e:
                    # Als ook CP1252 mislukt, sla het bestand over en waarschuw
                    st.warning(f"Kon {filename} niet lezen met UTF-8 of CP1252: {e}")
                    continue
            
            # Maak een LangChain Document object van de gelezen tekst
            if content is not None:
                doc = Document(page_content=content, metadata={'source': filepath})
                all_docs.append(doc)
    
    # Documenten verrijken met metadata en A2-samenvattingen
    docs_to_index = []
    for doc in all_docs:
        base_name = os.path.basename(doc.metadata['source'])
        meta_row = df_meta[df_meta['bestandsnaam'] == base_name]
        
        if not meta_row.empty:
            doc.metadata['volledige_tekst'] = doc.page_content # Sla volledige tekst op in metadata
            doc.page_content = str(meta_row.iloc[0]['a2_samenvatting']) # Indexeer op A2-samenvatting
            doc.metadata['bron'] = meta_row.iloc[0]['bron']
            doc.metadata['onderwerp'] = meta_row.iloc[0]['onderwerp']
            docs_to_index.append(doc)

    # Embeddings maken en Vector Store bouwen
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
    2.  Gebruik bullet points (•) voor de belangrijkste punten.
    3.  GEBRUIK DE VOLGENDE ICONEN VOOR DE BULLET POINTS:
        - 🏢 **Van wie:** Voor de afzender van de brief.
        - 🎯 **Wat moet u doen?:** Voor de belangrijkste actie die de gebruiker moet ondernemen.
        - 💰 **Bedrag:** Voor elk geldbedrag dat betaald moet worden of ontvangen wordt.
        - 🗓️ **Datum:** Voor elke belangrijke datum of deadline.
        - ℹ️ **Let op:** Voor overige belangrijke informatie.

    CONTEXT (VOLLEDIGE TEKST VAN DE BRIEF): {context}
    VRAAG: {question}
    HELPENDE ANTWOORD:
    """
)

# --- Hoofdapplicatie ---
st.set_page_config(page_title="Hulp bij Moeilijke Brieven", layout="wide")
st.title("🤖 AI Hulp voor Moeilijke Brieven")

# Controleer of Groq API key is ingesteld; anders stop de app.
if not groq_api_key:
    st.error("GROQ API sleutel niet gevonden. Zorg dat deze is ingesteld in je .env bestand of Streamlit Secrets.")
    st.stop()

# Initialiseer session state voor de levensduur van de app-sessie
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4()) # Unieke ID per sessie
if 'run_id' not in st.session_state:
    st.session_state.run_id = 0 # Teller voor unieke knop-keys per run

# Laden van kennisbank en initialisatie van LLM
with st.spinner("De intelligente kennisbank wordt geladen... Dit kan de eerste keer even duren."):
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

# --- Tab 1: Brief Uitleggen ---
with tab1:
    st.header("Laat je brief uitleggen")
    st.write("Plak de tekst van een moeilijke brief hieronder, of maak een foto van je brief.")
    
    # Checkbox voor akkoord privacy
    agreed_tab1 = st.checkbox("Ik begrijp dat mijn brief anoniem verwerkt wordt en ga akkoord.", key="agree_tab1", value=True)

    if agreed_tab1:
        # Kolommen voor OCR en tekstinvoer
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Optie 1: Maak een foto")
            uploaded_file = st.file_uploader("Kies een afbeelding...", type=["jpg", "png", "jpeg"], key="uploader_tab1")
            if uploaded_file:
                with st.spinner('Foto lezen...'):
                    try:
                        image = Image.open(uploaded_file)
                        text = pytesseract.image_to_string(image, lang='nld')
                        st.session_state.user_input_tab1 = text # Vul de session_state die de text_area voedt
                        st.success("Tekst uit foto gehaald!")
                    except Exception as e:
                         st.error(f"Fout bij het lezen van de afbeelding. Zorg dat Tesseract correct is geïnstalleerd. Details: {e}")
        
        with col2:
            st.subheader("Optie 2: Plak de tekst")
            # De text_area krijgt zijn waarde uit session_state, gevuld door OCR of directe input
            user_input = st.text_area("Tekst van de brief:", value=st.session_state.get('user_input_tab1', ''), height=300, key="input_tab1")

        # Knop om de brief uit te leggen
        if st.button("Leg de brief uit", type="primary", key="button_tab1"):
            st.session_state.run_id += 1 # Verhoog run_id voor unieke knoppen en mp3 bestandsnaam
            final_input = user_input # De tekst die de AI gaat verwerken
            
            if final_input:
                with st.spinner("Ik zoek in mijn kennisbank en maak een simpele samenvatting..."):
                    try:
                        response = qa_chain.invoke(final_input)
                        answer = response.get('result', '')
                        source_docs = response.get('source_documents', [])
                        
                        # Haal metadata op voor logging, als er documenten zijn gevonden
                        doc_metadata = source_docs[0].metadata if source_docs else {}

                        if answer:
                            st.subheader("Simpele Uitleg:")
                            st.markdown(f"<div>{answer}</div>", unsafe_allow_html=True) # Toon de uitleg
                            
                            # TTS - Voorleesfunctie
                            st.subheader("Lees de uitleg voor:")
                            tts_gebruikt_flag = False
                            try:
                                tts = gTTS(text=answer, lang='nl', slow=False)
                                tts_file = f"uitleg_{st.session_state.run_id}.mp3" 
                                tts.save(tts_file)
                                st.audio(tts_file)
                                tts_gebruikt_flag = True
                            except Exception as e:
                                st.error(f"Kon de audio niet aanmaken: {e}")
                            
                            # Feedback-knoppen
                            st.write("---")
                            st.write("**Was deze uitleg nuttig?**")
                            feedback_cols = st.columns(5)
                            with feedback_cols[0]:
                                if st.button("👍 Ja", key=f"yes_feedback_{st.session_state.run_id}"):
                                    log_to_supabase(st.session_state.session_id, doc_metadata.get('bron', 'N.v.t.'), doc_metadata.get('onderwerp', 'N.v.t.'), 1, tts_gebruikt=tts_gebruikt_flag)
                                    st.success("Bedankt voor je feedback!")
                            with feedback_cols[1]:
                                if st.button("👎 Nee", key=f"no_feedback_{st.session_state.run_id}"):
                                    log_to_supabase(st.session_state.session_id, doc_metadata.get('bron', 'N.v.t.'), doc_metadata.get('onderwerp', 'N.v.t.'), 0, tts_gebruikt=tts_gebruikt_flag)
                                    st.success("Bedankt voor je feedback!")
                            
                            # Expander voor gebruikte bronnen
                            with st.expander("Bekijk welke informatie is gebruikt"):
                                if source_docs:
                                    for doc in source_docs:
                                        st.info(f"**Gevonden document:** {doc.metadata.get('bron', 'Onbekend')} - {doc.metadata.get('onderwerp', 'Onbekend')}\n\n**Reden voor match (samenvatting):** *{doc.page_content}*")
                                else:
                                    st.info("Geen specifieke brondocumenten gevonden die de AI direct heeft gebruikt om dit antwoord te formuleren. Dit kan gebeuren bij irrelevante teksten.")
                        else:
                            # Log bij geen antwoord
                            log_to_supabase(st.session_state.session_id, "N.v.t.", "N.v.t.", -1, foutmelding="Geen antwoord van AI")
                            st.error("Het is niet gelukt een samenvatting te maken.")
                    except Exception as e:
                        # Log bij technische fout
                        log_to_supabase(st.session_state.session_id, "N.v.t.", "N.v.t.", -1, foutmelding=str(e))
                        st.error(f"Er is een technische fout opgetreden: {e}")
            else:
                st.warning("Plak eerst tekst of upload een foto van een brief.")

# --- Tab 2: Schrijfhulp ---
with tab2:
    st.header("Maak een professionele brief in simpele stappen")
    st.write("Kies welk soort brief je wilt sturen en vul de details in. De AI schrijft dan een voorbeeld voor je.")

    brief_type = st.selectbox(
        "**Stap 1: Kies wat voor soort brief je wilt schrijven**",
        ["--- Kies een optie ---", "Vraag om uitstel van betaling", "Bezwaar maken tegen een boete of beslissing",
         "Een afspraak afzeggen of verzetten", "Een abonnement of contract opzeggen"],
        key="brief_type_select"
    )
    if brief_type != "--- Kies een optie ---":
        st.write("---")
        st.subheader("**Stap 2: Vul de details in**")
        ontvanger = st.text_input("Aan wie is de brief? (Naam bedrijf of persoon)", key="ontvanger_input")
        kenmerk = st.text_input("Wat is het kenmerk, factuurnummer of klantnummer?", key="kenmerk_input")
        
        if "uitstel" in brief_type: extra_info_prompt = "Waarom vraag je uitstel en wanneer kun je wel betalen?"
        elif "Bezwaar" in brief_type: extra_info_prompt = "Waarom ben je het niet eens met de boete of beslissing?"
        elif "afspraak" in brief_type: extra_info_prompt = "Over welke afspraak gaat het (datum en tijd)?"
        elif "abonnement" in brief_type: extra_info_prompt = "Per wanneer wil je opzeggen?"
        else: extra_info_prompt = "Extra informatie:"
        extra_info = st.text_area(extra_info_prompt, key=f"extra_info_{brief_type}")

        if st.button("Schrijf mijn voorbeeldbrief", type="primary", key="button_tab2"):
            if ontvanger and kenmerk:
                schrijf_prompt_template = f"Schrijf een korte, beleefde en formele Nederlandse brief (B1-niveau). Doel: '{brief_type}'. Aan: {ontvanger}. Kenmerk: {kenmerk}. Extra info: '{extra_info}'. Zorg voor een correcte aanhef en afsluiting."
                with st.spinner("Ik schrijf een voorbeeldbrief..."):
                    try:
                        response = llm.invoke(schrijf_prompt_template)
                        st.subheader("Jouw voorbeeldbrief:")
                        st.text_area("Je kunt deze tekst kopiëren:", value=response.content, height=400, key="result_brief")
                    except Exception as e:
                        st.error(f"Er is iets misgegaan bij het schrijven van de brief. Fout: {e}")
            else:
                st.warning("Vul de naam van de ontvanger en het kenmerk in.")

# --- Privacyverklaring ---
st.markdown("---")
with st.expander("Privacy en Veiligheid"):
    st.write("""
    **Jouw privacy is belangrijk.**
    - Wij slaan jouw brieven en foto's **NIET** op.
    - Om de uitleg te geven, sturen we de tekst anoniem naar een beveiligde AI-dienst (Groq).
    - Deze dienst gebruikt jouw gegevens **niet** om hun AI te trainen.
    - Wij slaan wel **anonieme** feedback op (een 'duimpje omhoog' of 'omlaag') om de tool te verbeteren.
    """)