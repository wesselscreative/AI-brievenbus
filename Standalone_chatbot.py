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

# --- IMPORTS (alleen LLMChain is nu nodig) ---
from langchain.chains.llm import LLMChain

# --- Configuratie & Setup (NU CORRECT) ---
load_dotenv()
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
supabase_url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
supabase_key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))

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

# --- Prompts ---
PROMPT_UITLEG = PromptTemplate(
    input_variables=["context", "question"],
    template="""Je bent een empathische en zeer nauwkeurige AI-assistent die laaggeletterde mensen helpt. Je taak is om de onderstaande officiële brief te analyseren en in perfect, correct Nederlands (A2-niveau) uit te leggen.

STAP 1: ANALYSEER DE TOON
Lees de hele brief en bepaal of het goed, slecht, zeer ernstig of neutraal nieuws is.

STAP 2: SCHRIJF JE ANTWOORD
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
    * ℹ️ **Let op:** Vermeld hier andere belangrijke details.

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

# --- Callback Functies ---
def process_brief():
    if st.session_state.user_input:
        st.session_state.show_result = True
        st.session_state.feedback_given = False
        with st.spinner("Ik zoek relevante informatie en maak een samenvatting..."):
            try:
                retrieved_docs = retriever.get_relevant_documents(st.session_state.user_input)
                context_text = "\n\n".join([doc.metadata.get('volledige_tekst', doc.page_content) for doc in retrieved_docs])
                response = llm_chain_summary.invoke({
                    "context": context_text,
                    "question": st.session_state.user_input
                })
                summary_text = response.get('text', 'Kon geen samenvatting maken.')
                st.session_state.ai_response = { "result": summary_text, "source_documents": retrieved_docs }
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
    st.session_state.messages = []
    st.session_state.current_brief_text = ""
    st.session_state.current_summary = ""
    st.rerun()

# --- Hoofdapplicatie ---
st.set_page_config(page_title="Hulp bij Moeilijke Brieven", layout="wide")
st.title("🤖 AI Hulp voor Moeilijke Brieven")

if 'session_id' not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
if 'show_result' not in st.session_state: st.session_state.show_result = False
if 'feedback_given' not in st.session_state: st.session_state.feedback_given = False
if 'user_input' not in st.session_state: st.session_state.user_input = ""
if 'ai_response' not in st.session_state: st.session_state.ai_response = None
if 'messages' not in st.session_state: st.session_state.messages = []
if 'current_brief_text' not in st.session_state: st.session_state.current_brief_text = ""
if 'current_summary' not in st.session_state: st.session_state.current_summary = ""

if not groq_api_key or not supabase_url or not supabase_key:
    st.error("API sleutels niet gevonden.")
    st.stop()

with st.spinner("Kennisbank laden..."):
    vectorstore = load_and_process_data()
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=LLM_MODEL_GROQ)
    retriever = vectorstore.as_retriever()
    llm_chain_summary = LLMChain(llm=llm, prompt=PROMPT_UITLEG)

tab1, tab2 = st.tabs(["✉️ **Brief Laten Uitleggen**", "✍️ **Help Mij Schrijven**"])

with tab1:
    if not st.session_state.show_result:
        st.header("Laat je brief uitleggen")
        agreed = st.checkbox("Ik begrijp dat mijn brief anoniem wordt verwerkt.", value=True)
        if agreed:
            uploaded_file = st.file_uploader("Upload foto of PDF", type=["jpg", "png", "jpeg", "pdf"], label_visibility="collapsed")
            st.text_area("Plak de tekst", key="user_input", height=250, label_visibility="collapsed")
            st.button("Leg de brief uit", type="primary", on_click=process_brief)

if st.session_state.show_result:
    st.header("Uitleg en gesprek")
    chat_container = st.container(height=300, border=False)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    st.markdown("---")
    
    if len(st.session_state.messages) == 1:
        st.subheader("Laat de uitleg voorlezen:")
        try:
            tts = gTTS(text=st.session_state.current_summary, lang='nl')
            tts.save("uitleg.mp3")
            st.audio("uitleg.mp3")
        except Exception as e:
            st.warning(f"Voorlezen is nu niet beschikbaar. Fout: {e}")
        st.markdown("---")

    if not st.session_state.feedback_given:
        st.write("**Heeft de *eerste samenvatting* je geholpen?**")
        cols = st.columns(2)
        if cols[0].button("👍 Ja", use_container_width=True):
            handle_feedback(1)
            st.rerun()
        if cols[1].button("👎 Nee", use_container_width=True):
            handle_feedback(0)
            st.rerun()
    else:
        st.success("Bedankt voor je feedback!")

    if prompt := st.chat_input("Stel hier uw vraag over de brief..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Even denken..."):
            history_str = ""
            for msg in st.session_state.messages[:-1]:
                role = "Gebruiker" if msg['role'] == 'user' else 'Assistent'
                history_str += f"{role}: {msg['content']}\n"
            
            chat_chain = LLMChain(llm=llm, prompt=PROMPT_CHAT)
            
            response = chat_chain.invoke({
                "history": history_str,
                "input": prompt,
                "original_brief": st.session_state.current_brief_text,
                "summary": st.session_state.current_summary
            })
            
            ai_response_text = response.get('text', "Sorry, ik kan geen antwoord genereren.")
            st.session_state.messages.append({"role": "assistant", "content": ai_response_text})
        st.rerun()
        
    st.markdown("---")
    st.button("Begin opnieuw met een nieuwe brief", on_click=reset_app_state, use_container_width=True)

with tab2:
    st.header("Maak een professionele brief")
    st.write("Kies welk soort brief je wilt sturen en vul de details in.")
    brief_type = st.selectbox("**Stap 1: Soort brief**", ["--- Kies een optie ---", "Vraag om uitstel van betaling", "Bezwaar maken", "Afspraak afzeggen", "Abonnement opzeggen", "Sollicitatiebrief", "Klacht indienen"], key="brief_type_select")
    toon_keuze = st.selectbox("**Stap 2: Toon**", ["Zakelijk en formeel", "Vriendelijk maar dringend", "Neutraal en informatief", "Streng en direct", "Zeer boos en ontevreden"], key="toon_select")
    if brief_type != "--- Kies een optie ---":
        st.write("---")
        st.subheader("**Stap 3: Details**")
        ontvanger = st.text_input("Aan wie?", key="ontvanger_input")
        kenmerk = st.text_input("Kenmerk (factuurnummer, etc.)?", key="kenmerk_input")
        if "uitstel" in brief_type: extra_info_prompt = "Waarom vraag je uitstel en wanneer kun je wel betalen?"
        elif "Bezwaar" in brief_type: extra_info_prompt = "Waarom ben je het niet eens?"
        elif "afspraak" in brief_type: extra_info_prompt = "Welke afspraak (datum/tijd)? Wanneer zou je wel kunnen?"
        elif "abonnement" in brief_type: extra_info_prompt = "Per wanneer wil je opzeggen?"
        elif "Sollicitatiebrief" in brief_type: extra_info_prompt = "Voor welke functie? Wat zijn je vaardigheden?"
        elif "Klacht" in brief_type: extra_info_prompt = "Waarover gaat uw klacht? Wat is de gewenste oplossing?"
        else: extra_info_prompt = "Extra informatie:"
        extra_info = st.text_area(extra_info_prompt, key=f"extra_info_{brief_type.replace(' ', '_')}")
        if st.button("Schrijf mijn voorbeeldbrief", type="primary", key="button_tab2"):
            if ontvanger:
                schrijf_prompt = f"""Schrijf een Nederlandse brief op B1-niveau.
                - Doel: '{brief_type}'
                - Aan: {ontvanger}
                - Kenmerk: {kenmerk}
                - Toon: '{toon_keuze}'
                - Extra info: '{extra_info}'
                Zorg voor een correcte aanhef, logische structuur, en formele afsluiting."""
                with st.spinner("Ik schrijf een voorbeeldbrief..."):
                    response = llm.invoke(schrijf_prompt)
                    st.subheader("Jouw voorbeeldbrief:")
                    st.text_area("Je kunt deze tekst kopiëren en aanpassen:", value=response.content, height=400, key="result_brief")
            else:
                st.warning("Vul de naam van de ontvanger in.")

st.markdown("---")
with st.expander("Privacy en Veiligheid"):
    st.write("Jouw privacy is belangrijk. Wij slaan geen persoonlijke brieven op en gebruiken je data niet om de AI te trainen. We slaan alleen anonieme feedback op (ja/nee) om de tool te verbeteren.")