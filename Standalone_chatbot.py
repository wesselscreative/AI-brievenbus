import streamlit as st
import os
import pandas as pd
import uuid
import fitz  # PyMuPDF
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from gtts import gTTS
from supabase import create_client, Client

# --- NIEUWE IMPORTS VOOR DE ROBUUSTE CHAIN ---
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
# --- EINDE NIEUWE IMPORTS ---

# --- Configuratie & Setup ---
load_dotenv()
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
supabase_url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
supabase_key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "Data")
METADATA_PATH = os.path.join(SCRIPT_DIR, "metadata.csv")

LLM_MODEL_GROQ = "llama3-70b-8192"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Supabase Initialisatie ---
if supabase_url and supabase_key:
    supabase: Client = create_client(supabase_url, supabase_key)
else:
    supabase = None

# --- Functies (onveranderd) ---
def log_to_supabase(log_data):
    if not supabase: return
    try:
        supabase.table("logs").insert(log_data).execute()
        st.toast("Feedback opgeslagen!", icon="👍")
    except Exception as e:
        print(f"Fout bij het loggen naar Supabase: {e}")
        st.toast(f"Fout bij opslaan feedback.", icon="🔥")

@st.cache_resource
def load_and_process_data():
    try:
        df_meta = pd.read_csv(METADATA_PATH)
    except FileNotFoundError:
        st.error(f"Het bestand 'metadata.csv' is niet gevonden. De app kan niet starten.")
        st.stop()
    
    all_docs = []
    if not os.path.exists(DATA_PATH):
        st.error(f"De map 'Data' is niet gevonden. De app kan niet starten.")
        st.stop()
        
    for filename in os.listdir(DATA_PATH):
        filepath = os.path.join(DATA_PATH, filename)
        content = None
        try:
            if filename.lower().endswith(".pdf"):
                with fitz.open(filepath) as doc: content = "".join(page.get_text() for page in doc)
                if content: all_docs.append(Document(page_content=content, metadata={'source': filepath}))
            elif filename.lower().endswith(".txt"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f: content = f.read()
                except UnicodeDecodeError:
                    with open(filepath, 'r', encoding='cp1252') as f: content = f.read()
                if content is not None: all_docs.append(Document(page_content=content, metadata={'source': filepath}))
        except Exception as e: st.warning(f"Kon bestand {filename} niet laden: {e}")
    
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

# --- De Volledige Prompt ---
PROMPT_UITLEG = PromptTemplate.from_template(
    """Je bent een empathische en zeer nauwkeurige AI-assistent die laaggeletterde mensen helpt. Je taak is om de onderstaande officiële brief te analyseren en in perfect A2-taalgebruik uit te leggen.

STAP 1: ANALYSEER DE TOON
Lees eerst de hele brief en bepaal of het goed, slecht, zeer ernstig of neutraal nieuws is.

STAP 2: SCHRIJF JE ANTWOORD
Structureer je antwoord als volgt:

1.  **Introductiezin:** Begin met een zin die past bij de toon die je in Stap 1 hebt bepaald.
    - Bij goed nieuws: "Goed nieuws! In deze brief staat dat u..."
    - Bij slecht nieuws: "Dit is een belangrijke brief van [organisatie]. Hierin staat dat u..."
    - Bij zeer ernstig nieuws: "Let op, dit is een zeer dringende brief. Het is belangrijk dat u direct actie onderneemt. In de brief staat dat..."
    - Bij neutraal nieuws: "Dit is een brief met informatie over uw..."

2.  **Bullet Points:** Geef de details in een lijst. Elke bullet point MOET op een nieuwe regel beginnen met `* ` en het juiste icoon.
    * 🏢 **Van wie:** Noem de volledige naam van de afzender.
    * 🎯 **Wat moet u doen?:** Beschrijf de actie duidelijk. Als er geen actie nodig is, zeg dan "U hoeft niets te doen. Dit is ter informatie."
    * 💰 **Bedrag:** Noem ALLE bedragen met context. Bv: "U moet € 88 motorrijtuigenbelasting betalen."
    * 🗓️ **Datum:** Noem ALLE data met context. Bv: "U moet dit betalen vóór 22 november 2023."
    * ℹ️ **Let op:** Vermeld hier andere belangrijke details.

3.  **Afsluitende Vraag:** Eindig altijd met een open, helpende vraag, zoals: "Is dit zo duidelijk, of is er een woord dat ik extra moet uitleggen?"

ANALYSEER DE VOLGENDE CONTEXT (VOLLEDIGE TEKST VAN DE BRIEF):
{context}

VRAAG (NEGEER DEZE, FOCUS OP DE CONTEXT):
{question}

HELPENDE ANTWOORD:
"""
)

# --- Callback Functies (onveranderd) ---
def process_brief():
    if st.session_state.user_input:
        st.session_state.show_result = True
        st.session_state.feedback_given = False
        with st.spinner("Ik analyseer de brief en maak een simpele samenvatting..."):
            try:
                # De 'invoke' methode geeft de input als een dictionary mee
                st.session_state.ai_response = qa_chain.invoke({"query": st.session_state.user_input})
            except Exception as e:
                st.error(f"Er ging iets mis tijdens de analyse: {e}")
                st.session_state.ai_response = None
    else:
        st.warning("Plak eerst tekst in het vak of upload een bestand.")

def handle_feedback(score):
    if 'ai_response' in st.session_state and st.session_state.ai_response:
        response = st.session_state.ai_response
        source_docs = response.get('source_documents', [])
        doc_metadata = source_docs[0].metadata if source_docs else {}
        log_data = {"session_id": st.session_state.session_id, "document_br": doc_metadata.get('bron'), "document_on": doc_metadata.get('onderwerp'), "feedback_score": score}
        log_to_supabase(log_data)
        st.session_state.feedback_given = True

def reset_app_state():
    st.session_state.show_result = False
    st.session_state.feedback_given = False
    st.session_state.ai_response = None
    st.session_state.user_input = ""
    st.experimental_rerun()

# --- Hoofdapplicatie ---
st.set_page_config(page_title="Hulp bij Moeilijke Brieven", layout="wide")
st.title("🤖 AI Hulp voor Moeilijke Brieven")

# Session state initialisatie... (onveranderd)
if 'session_id' not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
if 'show_result' not in st.session_state: st.session_state.show_result = False
if 'feedback_given' not in st.session_state: st.session_state.feedback_given = False
if 'user_input' not in st.session_state: st.session_state.user_input = ""
if 'ai_response' not in st.session_state: st.session_state.ai_response = None

if not groq_api_key or not supabase_url or not supabase_key:
    st.error("Een of meerdere API sleutels (Groq, Supabase) zijn niet ingesteld. De applicatie kan niet starten.")
    st.stop()

with st.spinner("De intelligente kennisbank wordt klaargezet..."):
    vectorstore = load_and_process_data()
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=LLM_MODEL_GROQ)
    
    # --- DE NIEUWE, ROBUUSTE OPBOUW VAN DE QA CHAIN ---
    # Stap 1: Definieer de LLMChain die de prompt en de LLM combineert
    llm_chain = LLMChain(llm=llm, prompt=PROMPT_UITLEG)

    # Stap 2: Definieer de "Stuff" chain, die documenten in de prompt zal proppen.
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context" # Vertelt de stuff chain expliciet waar de documenttekst moet komen
    )

    # Stap 3: Creëer de uiteindelijke RetrievalQA chain, met de stuff_chain als de combineer-stap
    qa_chain = RetrievalQA(
        retriever=vectorstore.as_retriever(),
        combine_documents_chain=stuff_chain, # Gebruik onze custom-built chain
        return_source_documents=True,
        input_key="query" # Specificeer de input key voor de .invoke methode
    )
    # --- EINDE NIEUWE QA CHAIN OPBOUW ---


# --- UI met Tabs (onveranderd) ---
tab1, tab2 = st.tabs(["✉️ **Brief Laten Uitleggen**", "✍️ **Help Mij Schrijven**"])

# ... (De rest van de code voor de UI in tab1 en tab2, en de resultaatweergave blijft EXACT hetzelfde) ...
with tab1:
    st.header("Laat je brief uitleggen")
    agreed = st.checkbox("Ik begrijp dat mijn geüploade brief anoniem wordt verwerkt.", value=True)
    if agreed:
        uploaded_file = st.file_uploader("Optie 1: Upload een foto of PDF", type=["jpg", "png", "jpeg", "pdf"])
        if uploaded_file:
            with st.spinner('Bestand wordt gelezen...'):
                try:
                    if uploaded_file.name.lower().endswith('.pdf'):
                        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc: text_from_file = "".join(page.get_text() for page in doc)
                    else:
                        image = Image.open(uploaded_file)
                        text_from_file = pytesseract.image_to_string(image, lang='nld')
                    if text_from_file:
                        st.session_state.user_input = text_from_file
                        st.success("Het bestand is succesvol gelezen en de tekst staat hieronder.")
                    else:
                        st.warning("Ik kon geen tekst vinden in het bestand. Probeer het opnieuw of plak de tekst handmatig.")
                except Exception as e:
                    st.error(f"Fout bij het verwerken van het bestand: {e}")
        st.text_area("Optie 2: Plak de tekst (of controleer/bewerk de tekst van je upload)", key="user_input", height=250, help="Plak hier de volledige tekst van de brief.")
        col1, col2 = st.columns([1, 4])
        with col1:
            st.button("Leg de brief uit", type="primary", on_click=process_brief, disabled=st.session_state.show_result)

if st.session_state.show_result:
    st.markdown("---")
    if st.session_state.ai_response:
        response = st.session_state.ai_response
        answer = response.get('result', 'Sorry, er is iets misgegaan. Probeer het opnieuw.')
        
        st.subheader("Simpele Uitleg van je Brief:")
        st.markdown(answer, unsafe_allow_html=True)
        
        st.subheader("Laat de uitleg voorlezen:")
        try:
            tts = gTTS(text=answer, lang='nl')
            tts.save("uitleg.mp3")
            st.audio("uitleg.mp3")
        except Exception as e:
            st.warning(f"Het voorlezen is nu niet beschikbaar. Fout: {e}")
        
        st.write("---")
        if not st.session_state.feedback_given:
            st.write("**Heeft deze uitleg je geholpen?**")
            cols = st.columns(2)
            cols[0].button("👍 Ja, dit was nuttig", on_click=handle_feedback, args=(1,), use_container_width=True)
            cols[1].button("👎 Nee, dit was niet duidelijk", on_click=handle_feedback, args=(0,), use_container_width=True)
        else:
            st.success("Bedankt voor je feedback!")
        with st.expander("Bekijk welke informatie is gebruikt voor de samenvatting"):
            source_docs = response.get('source_documents', [])
            if source_docs:
                for doc in source_docs: st.info(f"**Gevonden document:** Bron: '{doc.metadata.get('bron')}', Onderwerp: '{doc.metadata.get('onderwerp')}'")
            else:
                st.write("Er is geen specifiek brondocument uit de database gebruikt.")
    else:
        st.warning("De analyse kon niet worden voltooid. Controleer de ingevoerde tekst en probeer het opnieuw.")
    st.button("Begin opnieuw met een nieuwe brief", on_click=reset_app_state, use_container_width=True)

with tab2:
    st.header("Maak een professionele brief in simpele stappen")
    st.write("Kies welk soort brief je wilt sturen en vul de details in. De AI schrijft dan een voorbeeld voor je.")
    
    brief_type = st.selectbox(
        "**Stap 1: Kies wat voor soort brief je wilt schrijven**",
        [
            "--- Kies een optie ---", 
            "Vraag om uitstel van betaling", 
            "Bezwaar maken tegen een boete of beslissing", 
            "Een afspraak afzeggen of verzetten", 
            "Een abonnement of contract opzeggen",
            "Sollicitatiebrief", 
            "Klacht indienen"
        ],
        key="brief_type_select"
    )
    
    if brief_type != "--- Kies een optie ---":
        st.write("---")
        st.subheader("**Stap 2: Vul de details in**")
        ontvanger = st.text_input("Aan wie is de brief? (Naam bedrijf of persoon)", key="ontvanger_input")
        kenmerk = st.text_input("Wat is het kenmerk, factuurnummer, vacaturenummer of klantnummer?", key="kenmerk_input")
        
        if "uitstel" in brief_type: extra_info_prompt = "Waarom vraag je uitstel en wanneer kun je wel betalen?"
        elif "Bezwaar" in brief_type: extra_info_prompt = "Waarom ben je het niet eens met de boete of beslissing?"
        elif "afspraak" in brief_type: extra_info_prompt = "Over welke afspraak gaat het (datum en tijd) en wanneer zou je wel kunnen?"
        elif "abonnement" in brief_type: extra_info_prompt = "Per wanneer wil je opzeggen? Vermeld eventueel de reden."
        elif "Sollicitatiebrief" in brief_type: extra_info_prompt = "Voor welke functie en bij welk bedrijf solliciteer je? Wat zijn je belangrijkste vaardigheden voor deze baan?"
        elif "Klacht" in brief_type: extra_info_prompt = "Waarover gaat uw klacht (product/dienst)? Wat is er misgegaan en welke oplossing wilt u?"
        else: extra_info_prompt = "Extra informatie:"
        
        extra_info = st.text_area(extra_info_prompt, key=f"extra_info_{brief_type.replace(' ', '_')}")
        
        if st.button("Schrijf mijn voorbeeldbrief", type="primary", key="button_tab2"):
            if ontvanger and kenmerk:
                schrijf_prompt = f"Schrijf een korte, beleefde en formele Nederlandse brief op B1-niveau. Doel: '{brief_type}'. Aan: {ontvanger}. Kenmerk: {kenmerk}. Gebruik deze extra informatie van de gebruiker: '{extra_info}'. Zorg voor een correcte aanhef, structuur met alinea's, en een formele afsluiting."
                with st.spinner("Ik schrijf een voorbeeldbrief..."):
                    response = llm.invoke(schrijf_prompt)
                    st.subheader("Jouw voorbeeldbrief:")
                    st.text_area("Je kunt deze tekst kopiëren en aanpassen:", value=response.content, height=400, key="result_brief")
            else:
                st.warning("Vul in ieder geval de naam van de ontvanger en het kenmerk in.")

st.markdown("---")
with st.expander("Privacy en Veiligheid"):
    st.write("Jouw privacy is belangrijk. Wij slaan geen persoonlijke brieven op en gebruiken je data niet om de AI te trainen. We slaan alleen anonieme feedback op (ja/nee) om de tool te verbeteren.")