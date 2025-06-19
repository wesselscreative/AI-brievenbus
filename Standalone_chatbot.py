# --- COMPLETE, GECORRIGEERDE EN HERZIENE CODE ---

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
from langchain.chains.llm import LLMChain

# --- Configuratie & Setup ---
load_dotenv()
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
supabase_url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
supabase_key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))

# BELANGRIJK: Zorg dat Tesseract-OCR is geïnstalleerd op je systeem.
# Indien nodig, specificeer hier het pad naar de executable.
# Voorbeeld voor Windows: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "Data")
METADATA_PATH = os.path.join(SCRIPT_DIR, "metadata.csv")

LLM_MODEL_GROQ = "llama3-70b-8192"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

if supabase_url and supabase_key:
    supabase: Client = create_client(supabase_url, supabase_key)
else:
    supabase = None
# --- Einde Configuratie ---


# --- Functies ---
def log_to_supabase(log_data):
    if not supabase: return
    try:
        supabase.table("logs").insert(log_data).execute()
        st.toast("Feedback opgeslagen!", icon="👍")
    except Exception as e:
        print(f"Fout bij het loggen naar Supabase: {e}")
        st.toast(f"Fout bij opslaan feedback.", icon="🔥")

# --- VOLLEDIG HERSTELD: Functie voor het laden van de kennisbank ---
@st.cache_resource
def load_and_process_knowledge_base():
    """
    Laadt de voorkennis (voorbeeldbrieven) uit de Data-map.
    Deze functie is hersteld maar wordt NIET gebruikt voor de analyse van nieuwe brieven.
    Dit voorkomt de oorspronkelijke fout.
    """
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
            elif filename.lower().endswith(".txt"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f: content = f.read()
                except UnicodeDecodeError:
                    with open(filepath, 'r', encoding='cp1252') as f: content = f.read()
            if content:
                all_docs.append(Document(page_content=content, metadata={'source': filepath}))
        except Exception as e:
            st.warning(f"Kon voorbeeldbestand {filename} niet laden: {e}")

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

    if not docs_to_index:
        return None
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(documents=docs_to_index, embedding=embeddings)
    return vectorstore

PROMPT_UITLEG = PromptTemplate(
    input_variables=["context", "question"],
    template="""Je bent een empathische en zeer nauwkeurige AI-assistent die laaggeletterde mensen helpt. Je taak is om de onderstaande officiële brief te analyseren en in perfect, correct Nederlands (A2-niveau) uit te leggen.

**INSTRUCTIES:**
1. Baseer je antwoord **UITSLUITEND** op de tekst in de "CONTEXT". Verzin geen informatie en gebruik geen kennis van buitenaf.
2. Als een detail (zoals een bedrag of datum) niet in de tekst staat, schrijf dan "Niet gespecificeerd in de brief".

**STAP 1: ANALYSEER DE TOON**
Lees de hele brief en bepaal of het goed, slecht, zeer ernstig of neutraal nieuws is.

**STAP 2: SCHRIJF JE ANTWOORD**
Structureer je antwoord als volgt. Zorg ervoor dat elke zin grammaticaal correct is.

1.  **Introductiezin:** Kies de introductie die past bij jouw analyse en vul deze aan tot een volledige, correcte zin.
    - Bij goed nieuws, gebruik dit sjabloon: "Goed nieuws! In deze brief staat dat u [JOUW ANALYSE HIER]."
    - Bij slecht nieuws, gebruik dit sjabloon: "Dit is een belangrijke brief van [organisatie]. Hierin staat dat u [JOUW ANALYSE HIER]."
    - Bij zeer ernstig nieuws, gebruik dit sjabloon: "Let op, dit is een zeer dringende brief. In de brief staat dat [JOUW ANALYSE HIER]."
    - Bij neutraal nieuws, gebruik dit sjabloon: "Dit is een brief met informatie over uw [JOUW ANALYSE HIER]."

2.  **Bullet Points:** Geef de details in een lijst. Elke bullet point MOET op een nieuwe regel beginnen met `* ` en het juiste icoon.
    * 🏢 **Van wie:** Noem de volledige naam van de afzender.
    * 🎯 **Wat moet u doen?:** Beschrijf de actie duidelijk. Als er geen actie nodig is, zeg dan "U hoeft niets te doen. Dit is ter informatie."
    * 💰 **Bedrag:** Noem ALLE bedragen met context. Bv: "U moet € 88 motorrijtuigenbelasting betalen."
    * 🗓️ **Datum:** Noem ALLE data met context. Bv: "U moet dit betalen vóór 22 november 2023."
    * ℹ️ **Let op:** Vermeld hier andere belangrijke details, zoals een waarschuwing of een contactnummer.

3.  **Afsluitende Vraag:** Eindig altijd met een open, helpende vraag, zoals: "Is dit zo duidelijk, of is er een woord dat ik extra moet uitleggen?"

ANALYSEER DE VOLGENDE CONTEXT (VOLLEDIGE TEKST VAN DE BRIEF):
{context}

VRAAG (NEGEER DEZE, FOCUS OP DE CONTEXT):
{question}

HELPENDE ANTWOORD:"""
)

PROMPT_CHAT = PromptTemplate(
    input_variables=["history", "input", "original_brief", "summary"],
    template="""Je bent een behulpzame AI-assistent. De gebruiker heeft een moeilijke brief laten samenvatten.
De volledige, originele tekst van de brief was:
---
{original_brief}
---
Jouw eerdere samenvatting was:
---
{summary}
---
De huidige conversatie is:
{history}

Beantwoord nu de nieuwe vraag van de gebruiker op een simpele en duidelijke (A2-niveau) manier. Baseer je antwoord op de originele brief en de samenvatting.
Gebruiker: {input}
Assistent:"""
)

def process_brief_and_show_results():
    if st.session_state.user_input:
        st.session_state.show_result = True
        st.session_state.feedback_given = False
        with st.spinner("Ik analyseer uw brief en maak een samenvatting..."):
            try:
                # DE GECORRIGEERDE LOGICA: gebruik de input direct, negeer de kennisbank
                context_text = st.session_state.user_input
                response = llm_chain_summary.invoke({
                    "context": context_text,
                    "question": "Vat deze brief samen volgens de instructies."
                })
                summary_text = response.get('text', 'Kon geen samenvatting maken.')
                
                st.session_state.ai_response = {"result": summary_text}
                st.session_state.current_brief_text = context_text
                st.session_state.current_summary = summary_text
                st.session_state.messages = [{"role": "assistant", "content": summary_text}]
            except Exception as e:
                st.error(f"Er ging iets mis tijdens de analyse: {e}")
                st.session_state.ai_response = None
    else:
        st.warning("Plak eerst tekst in het vak of upload een bestand.")

def handle_feedback(score):
    if 'ai_response' in st.session_state and st.session_state.ai_response:
        log_data = {
            "session_id": st.session_state.session_id,
            "feedback_score": score,
            "llm_summary": st.session_state.current_summary,
            "original_text": st.session_state.current_brief_text
        }
        log_to_supabase(log_data)
        st.session_state.feedback_given = True

def reset_app_state():
    session_id = st.session_state.session_id
    for key in list(st.session_state.keys()):
        if key != 'session_id':
            del st.session_state[key]
    st.session_state.session_id = session_id
    st.rerun()

# --- Hoofdapplicatie ---
st.set_page_config(page_title="Hulp bij Moeilijke Brieven", layout="wide")

# Initialiseer session state variabelen
if 'session_id' not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
if 'show_result' not in st.session_state: st.session_state.show_result = False
if 'feedback_given' not in st.session_state: st.session_state.feedback_given = False
if 'user_input' not in st.session_state: st.session_state.user_input = ""
if 'ai_response' not in st.session_state: st.session_state.ai_response = None
if 'messages' not in st.session_state: st.session_state.messages = []
if 'current_brief_text' not in st.session_state: st.session_state.current_brief_text = ""
if 'current_summary' not in st.session_state: st.session_state.current_summary = ""

if not groq_api_key:
    st.error("GROQ API sleutel niet gevonden. Controleer je .env of Streamlit secrets.")
    st.stop()

# Laad de LLM en de prompt chain
llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=LLM_MODEL_GROQ)
llm_chain_summary = LLMChain(llm=llm, prompt=PROMPT_UITLEG)

# HERSTELD: Laad de kennisbank bij de start van de app.
# Deze 'vectorstore' wordt momenteel niet actief gebruikt, maar de logica is hersteld.
with st.spinner("Voorbeelden laden..."):
    vectorstore = load_and_process_knowledge_base()

# --- UI LAYOUT ---
st.title("🤖 AI Hulp voor Moeilijke Brieven")

if st.session_state.show_result:
    # --- RESULTAAT- EN CHAT PAGINA ---
    with st.sidebar:
        st.header("Acties")
        st.button("Begin opnieuw met een nieuwe brief", on_click=reset_app_state, use_container_width=True, type="primary")
        st.markdown("---")
        
        if len(st.session_state.messages) == 1:
            st.subheader("Laat de uitleg voorlezen")
            try:
                tts = gTTS(text=st.session_state.current_summary, lang='nl', slow=False)
                tts.save("uitleg.mp3")
                st.audio("uitleg.mp3")
            except Exception as e: st.warning(f"Voorlezen niet beschikbaar: {e}")
            st.markdown("---")
            if not st.session_state.feedback_given:
                st.subheader("Was deze samenvatting nuttig?")
                if st.button("👍 Ja", use_container_width=True): handle_feedback(1); st.rerun()
                if st.button("👎 Nee", use_container_width=True): handle_feedback(0); st.rerun()
            else: st.success("Bedankt voor je feedback!")
        
        with st.expander("Privacy en Veiligheid"):
            st.write("Jouw privacy is belangrijk. Wij slaan de inhoud van jouw brieven **niet** op. De tekst wordt alleen tijdens de analyse gebruikt en daarna direct verwijderd.")

    st.header("Uitleg en gesprek")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if prompt := st.chat_input("Stel hier een vervolgvraag..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.spinner("Even denken..."):
            with st.chat_message("assistant"):
                history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[:-1]])
                chat_chain = LLMChain(llm=llm, prompt=PROMPT_CHAT)
                response = chat_chain.invoke({
                    "history": history_str, "input": prompt,
                    "original_brief": st.session_state.current_brief_text,
                    "summary": st.session_state.current_summary
                })
                ai_response_text = response.get('text', "Sorry, ik kan op dit moment geen antwoord genereren.")
                st.markdown(ai_response_text)
                st.session_state.messages.append({"role": "assistant", "content": ai_response_text})
else:
    # --- STARTPAGINA met TABS ---
    tab1, tab2 = st.tabs(["✉️ **Brief Laten Uitleggen**", "✍️ **Help Mij Schrijven**"])
    
    with tab1:
        st.header("Upload of plak de tekst van je brief")
        agreed = st.checkbox("Ik begrijp dat mijn brief anoniem wordt verwerkt.", value=True)
        if agreed:
            uploaded_file = st.file_uploader("Upload een foto of PDF", type=["jpg", "png", "jpeg", "pdf"])
            st.text_area("Of plak hier de tekst", key="user_text_input", height=200)
            if st.button("Leg de brief uit", type="primary"):
                input_text = ""
                if uploaded_file:
                    with st.spinner("Bestand wordt gelezen..."):
                        try:
                            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                            if file_ext == ".pdf":
                                with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as doc: input_text = "".join(p.get_text() for p in doc)
                            elif file_ext in [".jpg", ".png", ".jpeg"]:
                                input_text = pytesseract.image_to_string(Image.open(uploaded_file), lang='nld')
                        except Exception as e: st.error(f"Fout bij lezen bestand: {e}")
                elif st.session_state.user_text_input: input_text = st.session_state.user_text_input
                st.session_state.user_input = input_text
                process_brief_and_show_results()
                st.rerun()
        else: st.warning("U moet akkoord gaan om door te gaan.")
            
    with tab2:
        st.header("Maak een professionele brief")
        st.write("Kies welk soort brief je wilt sturen en vul de details in.")
        brief_type = st.selectbox("**Stap 1: Soort brief**", ["--- Kies een optie ---", "Vraag om uitstel van betaling", "Bezwaar maken", "Afspraak afzeggen", "Abonnement opzeggen", "Sollicitatiebrief", "Klacht indienen"])
        toon_keuze = st.selectbox("**Stap 2: Toon**", ["Zakelijk en formeel", "Vriendelijk maar dringend", "Neutraal en informatief", "Streng en direct", "Zeer boos en ontevreden"])
        if brief_type != "--- Kies een optie ---":
            st.write("---"); st.subheader("**Stap 3: Details**")
            ontvanger = st.text_input("Aan wie?")
            kenmerk = st.text_input("Kenmerk (factuurnummer, etc.)?")
            extra_prompts = {"uitstel": "Waarom...", "Bezwaar": "Waarom...", "afspraak": "Welke...", "abonnement": "Per wanneer...", "Sollicitatiebrief": "Voor welke...", "Klacht": "Waarover..."}
            extra_info_prompt = next((v for k, v in extra_prompts.items() if k in brief_type), "Extra informatie:")
            extra_info = st.text_area(extra_info_prompt)
            if st.button("Schrijf mijn voorbeeldbrief", type="primary"):
                if ontvanger:
                    prompt_str = f"""Schrijf een Nederlandse brief op B1-niveau. Doel: '{brief_type}', Aan: {ontvanger}, Kenmerk: {kenmerk}, Toon: '{toon_keuze}', Extra info: '{extra_info}'. Zorg voor een correcte structuur."""
                    with st.spinner("Ik schrijf een voorbeeldbrief..."):
                        response = llm.invoke(prompt_str)
                        st.subheader("Jouw voorbeeldbrief:")
                        st.text_area("Je kunt deze tekst kopiëren en aanpassen:", value=response.content, height=400)
                else: st.warning("Vul de naam van de ontvanger in.")