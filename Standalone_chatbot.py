# --- BENODIGDE PACKAGES ---
# Zorg ervoor dat je deze hebt geïnstalleerd:
# pip install streamlit pandas py-mu-pdf langchain langchain-community langchain-groq python-dotenv Pillow pytesseract python-gtts supabase-py faiss-cpu sentence-transformers

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
from io import BytesIO
from datetime import datetime
import re # Nodig voor het opschonen van tekst voor audio

# --- Configuratie & Setup ---
load_dotenv()
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
supabase_url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
supabase_key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))

# BELANGRIJK: Zorg dat Tesseract-OCR is geïnstalleerd op je systeem.
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

@st.cache_resource
def load_and_process_knowledge_base():
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
    if not docs_to_index: return None
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(documents=docs_to_index, embedding=embeddings)
    return vectorstore

PROMPT_UITLEG = PromptTemplate(
    input_variables=["context", "question"],
    template="""Je bent een empathische en zeer nauwkeurige AI-assistent die laaggeletterde mensen helpt. Je taak is om de onderstaande officiële brief te analyseren en in perfect, correct Nederlands (A2-niveau) uit te leggen.

**INSTRUCTIES:**
1. Baseer je antwoord **UITSLUITEND** op de tekst in de "CONTEXT". Verzin geen informatie.
2. Structureer je antwoord exact zoals hieronder beschreven.

**Analyseer de toon en schrijf je antwoord:**
Begin met een introductiezin die past bij de toon (goed, slecht, ernstig, neutraal nieuws).
Geef daarna de details in een lijst. Elke bullet point MOET op een nieuwe regel beginnen met `* ` en het juiste icoon.
    * 🏢 **Van wie:** Noem de volledige naam van de afzender.
    * 🎯 **Wat moet u doen?:** Beschrijf de actie duidelijk.
    * 💰 **Bedrag:** Noem ALLE bedragen met context.
    * 🗓️ **Datum:** Noem ALLE data met context.
    * ℹ️ **Let op:** Vermeld hier andere belangrijke details.
Eindig altijd met: "Is dit zo duidelijk, of is er een woord dat ik extra moet uitleggen?"

**Identificeer de vervolgactie (voor intern gebruik):**
Na je volledige antwoord, voeg een nieuwe regel toe die begint met `###ACTIE###`. Identificeer op basis van de brief de meest logische vervolgactie voor de gebruiker. Kies uit: `Betalen`, `Uitstel vragen`, `Bezwaar maken`, `Afspraak afzeggen`, `Abonnement opzeggen`, `Klacht indienen`, `Solliciteren`, `Geen actie nodig`.
Voorbeeld: ###ACTIE### Uitstel vragen

**Identificeer de data (voor intern gebruik):**
Na de actie-regel, voeg een nieuwe regel toe die begint met `###DATA###`. Extraheer de naam van de afzender en het kenmerk/dossiernummer. Formaat: `Afzender: [Naam] | Kenmerk: [Nummer]`. Als iets niet gevonden is, gebruik "N.v.t.".
Voorbeeld: ###DATA### Afzender: Intrum Justitia B.V. | Kenmerk: 10987654

ANALYSEER DE VOLGENDE CONTEXT:
{context}

VRAAG (NEGEER DEZE):
{question}

HELPENDE ANTWOORD:"""
)

PROMPT_CHAT = PromptTemplate(
    input_variables=["history", "input", "original_brief", "summary"],
    template="""Je bent een behulpzame AI-assistent. De gebruiker heeft een moeilijke brief laten samenvatten. Jouw taak is om vervolgvragen te beantwoorden.
**REGELS:**
1. **GEBRUIK ALLEEN DE ORIGINELE BRIEF:** Baseer je antwoord **uitsluitend** op de "Originele Tekst van de Brief".
2. **HOUD HET SIMPEL (A2-niveau):** Gebruik korte zinnen en makkelijke woorden.
3. **WEES DIRECT:** Geef antwoord op de vraag van de gebruiker.

Originele Tekst van de Brief:
---
{original_brief}
---
Jouw eerdere samenvatting (alleen ter context):
---
{summary}
---
De huidige conversatie is:
{history}

Beantwoord nu de nieuwe vraag van de gebruiker.
Gebruiker: {input}
Assistent:"""
)

PROMPT_GEVOLGEN = PromptTemplate(
    input_variables=["context"],
    template="""Je bent een rustige en realistische AI-adviseur voor mensen die moeite hebben met lezen. Jouw taak is om uit te leggen wat de gevolgen zijn van de onderstaande brief.
**INSTRUCTIES:**
1. Baseer je antwoord **UITSLUITEND** op de tekst in de "CONTEXT".
2. Gebruik zeer eenvoudige taal (A2-niveau). Korte zinnen.
3. Wees niet te alarmerend, maar wel eerlijk over de risico's.

**ANALYSEER DE VOLGENDE CONTEXT (VOLLEDIGE TEKST VAN DE BRIEF):**
{context}

**JOUW ANTWOORD:**
Schrijf een korte, duidelijke uitleg die de volgende vragen beantwoordt:
* **Wat gebeurt er als ik niets doe?** (Beschrijf de logische volgende stap, bv. "dan kan er een gerechtsdeurwaarder komen").
* **Wat is het beste wat ik nu kan doen?** (Geef de meest constructieve actie aan, bv. "het is het beste om direct contact op te nemen met [organisatie]").

Structureer je antwoord als een korte alinea. Bijvoorbeeld: "Als u niets doet, zullen de kosten waarschijnlijk hoger worden en kan er een deurwaarder worden ingeschakeld. Het beste wat u nu kunt doen, is contact opnemen met Intrum om te vragen of u in delen mag betalen."
"""
)

PROMPT_SCHRIJVEN_NIEUW = PromptTemplate(
    input_variables=["doel_brief", "ontvanger", "kenmerk", "toon", "extra_info"],
    template="""Je bent een expert in het schrijven van formele Nederlandse brieven. Je taak is om een perfect gestructureerde en foutloze conceptbrief te genereren op B1-taalniveau.

**INSTRUCTIES:**
1. Volg de onderstaande structuur **exact**.
2. Gebruik de placeholders zoals `[Jouw Naam]` waar de gebruiker zelf informatie moet invullen.
3. Formuleer de kern van de brief op basis van de input (`doel_brief`, `toon`, `extra_info`).
4. Wees altijd beleefd en professioneel, zelfs als de gevraagde toon 'boos' is. Vertaal 'boos' naar 'zeer dringend en ontevreden'.

**INPUTGEGEVENS:**
- **Doel:** {doel_brief}
- **Aan:** {ontvanger}
- **Kenmerk:** {kenmerk}
- **Toon:** {toon}
- **Extra Informatie van gebruiker:** {extra_info}

**BRIEFSTRUCTUUR (GEBRUIK DEZE EXACT):**

[Jouw Naam]
[Jouw Straat en Huisnummer]
[Jouw Postcode en Woonplaats]
[Jouw E-mailadres]
[Jouw Telefoonnummer]

{ontvanger}
[Adres van Ontvanger]
[Postcode en Plaats van Ontvanger]

[Jouw Woonplaats], {current_date}

**Betreft:** {kenmerk}

Geachte heer/mevrouw,

Ik schrijf u naar aanleiding van [BESCHRIJF HIER KORT DE AANLEIDING, BIJV. 'uw brief met kenmerk {kenmerk}'].

[HIER KOMT DE KERN VAN DE BRIEF. Verwerk hier de '{extra_info}' van de gebruiker. Formuleer duidelijke, korte alinea's. Pas de schrijfstijl aan op de gevraagde '{toon}'.]

[HIER KOMT DE AFSLUITENDE ALINEA. Beschrijf duidelijk wat je als volgende stap verwacht. Bijvoorbeeld: 'Ik zie uw reactie graag binnen 14 dagen tegemoet.' of 'Ik hoop op uw begrip en zie uit naar een positieve oplossing.']

Met vriendelijke groet,

[Jouw Naam]
"""
)

@st.cache_data
def generate_audio_from_text(text):
    """Genereert audio van de tekst en geeft de bytes terug. Wordt gecached."""
    try:
        # Opschonen van de tekst voor betere audio
        clean_text = re.sub(r'\*\*', '', text) # Verwijder **
        clean_text = re.sub(r'\*', '', clean_text)  # Verwijder *
        clean_text = re.sub(r'🏢|🎯|💰|🗓️|ℹ️', '', clean_text) # Verwijder iconen
        clean_text = re.sub(r'Van wie:', 'Van wie:', clean_text)
        clean_text = re.sub(r'Wat moet u doen\?:', 'Wat moet u doen?', clean_text)
        clean_text = re.sub(r'Bedrag:', 'Bedrag:', clean_text)
        clean_text = re.sub(r'Datum:', 'Datum:', clean_text)
        clean_text = re.sub(r'Let op:', 'Let op:', clean_text)
        
        tts = gTTS(text=clean_text, lang='nl', slow=False)
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp.getvalue()
    except Exception as e:
        print(f"Fout bij genereren audio: {e}")
        return None

def handle_feedback(score):
    if 'current_summary' in st.session_state and st.session_state.current_summary:
        log_data = {"session_id": st.session_state.session_id, "feedback_score": score, "llm_summary": st.session_state.current_summary, "original_text": st.session_state.current_brief_text}
        log_to_supabase(log_data)
        st.session_state.feedback_given = True

def reset_app_state():
    session_id = st.session_state.session_id
    keys_to_keep = ['session_id']
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    st.session_state.session_id = session_id
    st.rerun()

# --- Hoofdapplicatie ---
st.set_page_config(page_title="Hulp bij Moeilijke Brieven", layout="wide")

# Initialiseer session state variabelen
if 'session_id' not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
if 'show_result' not in st.session_state: st.session_state.show_result = False
if 'feedback_given' not in st.session_state: st.session_state.feedback_given = False
if 'messages' not in st.session_state: st.session_state.messages = []
if 'current_brief_text' not in st.session_state: st.session_state.current_brief_text = ""
if 'current_summary' not in st.session_state: st.session_state.current_summary = ""
if 'current_mode' not in st.session_state: st.session_state.current_mode = ""

if not groq_api_key:
    st.error("GROQ API sleutel niet gevonden. Controleer je .env of Streamlit secrets.")
    st.stop()

llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=LLM_MODEL_GROQ)
with st.spinner("Voorbeelden laden..."):
    vectorstore = load_and_process_knowledge_base()

st.title("🤖 AI Hulp voor Moeilijke Brieven")

if st.session_state.show_result:
    # --- RESULTAAT- EN CHAT PAGINA ---
    with st.sidebar:
        st.header("Acties")
        st.button("Begin opnieuw met een nieuwe brief", on_click=reset_app_state, use_container_width=True, type="primary")
        st.markdown("---")
        
        if len(st.session_state.messages) == 1 and st.session_state.current_mode == 'uitleggen':
            st.subheader("Laat de uitleg voorlezen")
            
            # De audio wordt nu gegenereerd van de originele samenvatting, de opschoningslogica zit in de functie zelf
            audio_bytes = generate_audio_from_text(st.session_state.current_summary)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
            else:
                st.warning("Voorlezen is op dit moment niet beschikbaar.")
            st.markdown("---")
            if not st.session_state.feedback_given:
                st.subheader("Was deze samenvatting nuttig?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Ja", use_container_width=True): handle_feedback(1); st.rerun()
                with col2:
                    if st.button("👎 Nee", use_container_width=True): handle_feedback(0); st.rerun()
            else: st.success("Bedankt voor je feedback!")
        
        with st.expander("Privacy en Veiligheid"):
            st.write("Jouw privacy is belangrijk. Wij slaan de inhoud van jouw brieven **niet** op. De tekst wordt alleen tijdens de analyse gebruikt en daarna direct verwijderd.")

    if st.session_state.current_mode == 'uitleggen':
        st.markdown("##### Modus: ✉️ **Brief wordt uitgelegd**")

    st.header("Uitleg en gesprek")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if len(st.session_state.messages) == 1:
        with st.expander("🤔 Wat betekent dit voor mij? Klik hier voor extra uitleg."):
            with st.spinner("Ik analyseer de gevolgen..."):
                llm_chain_gevolgen = LLMChain(llm=llm, prompt=PROMPT_GEVOLGEN)
                response = llm_chain_gevolgen.invoke({"context": st.session_state.current_brief_text})
                gevolgen_text = response.get('text', "Kon de gevolgen niet analyseren.")
                st.info(gevolgen_text)

    if len(st.session_state.messages) == 1 and st.session_state.get('suggested_action'):
        action_name = st.session_state.suggested_action
        if action_name != "Geen actie nodig":
            st.info(f"**Vervolgstap:** Het lijkt erop dat de beste actie is om een brief te sturen om **{action_name.lower()}**.")
            if st.button(f"Ja, help mij een brief schrijven voor '{action_name}'", type="primary"):
                st.session_state.prefill_action = True
                st.session_state.show_result = False
                st.rerun()

    # De chat input is nu alleen tekst, geen spraak
    if final_prompt := st.chat_input("Of typ hier je vraag..."):
        st.session_state.messages.append({"role": "user", "content": final_prompt})
        with st.chat_message("user"): st.markdown(final_prompt)
        with st.spinner("Even denken..."):
            with st.chat_message("assistant"):
                history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[:-1]])
                chat_chain = LLMChain(llm=llm, prompt=PROMPT_CHAT)
                response = chat_chain.invoke({"history": history_str, "input": final_prompt, "original_brief": st.session_state.current_brief_text, "summary": st.session_state.current_summary})
                ai_response_text = response.get('text', "Sorry, ik kan op dit moment geen antwoord genereren.")
                st.markdown(ai_response_text)
                st.session_state.messages.append({"role": "assistant", "content": ai_response_text})
        st.rerun()

else:
    # --- STARTPAGINA met TABS ---
    tab1, tab2 = st.tabs(["✉️ **Brief Laten Uitleggen**", "✍️ **Help Mij Schrijven**"])
    
    with tab1:
        st.header("Upload of plak de tekst van je brief")
        agreed = st.checkbox("Ik begrijp dat mijn brief anoniem wordt verwerkt.", value=True)
        if agreed:
            uploaded_file = st.file_uploader("Upload een foto of PDF", type=["jpg", "png", "jpeg", "pdf"])
            user_text_area = st.text_area("Of plak hier de tekst", key="user_text_input", height=200)
            
            if st.button("Leg de brief uit", type="primary"):
                input_text = ""
                if uploaded_file:
                    with st.spinner("Bestand wordt gelezen..."):
                        try:
                            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                            if file_ext == ".pdf":
                                with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as doc: 
                                    input_text = "".join(p.get_text() for p in doc)
                            elif file_ext in [".jpg", ".png", ".jpeg"]:
                                input_text = pytesseract.image_to_string(Image.open(uploaded_file), lang='nld')
                        except Exception as e: 
                            st.error(f"Fout bij het lezen van het bestand: {e}"); st.stop()
                elif user_text_area:
                    input_text = user_text_area

                if not input_text or not input_text.strip():
                    st.error("❌ Geen tekst gevonden. Upload een bestand of plak tekst in het vak.")
                else:
                    st.session_state.current_mode = 'uitleggen'
                    with st.spinner("Ik analyseer uw brief en maak een samenvatting..."):
                        try:
                            llm_chain_summary = LLMChain(llm=llm, prompt=PROMPT_UITLEG)
                            response = llm_chain_summary.invoke({"context": input_text, "question": "Vat deze brief samen."})
                            full_response_text = response.get('text', 'Kon geen samenvatting maken.')

                            if "###ACTIE###" in full_response_text:
                                parts = full_response_text.split("###ACTIE###")
                                summary_text = parts[0].strip()
                                action_part = parts[1]
                                st.session_state.suggested_action = action_part.split('\n')[0].strip()
                                if "###DATA###" in action_part:
                                    data_line = action_part.split("###DATA###")[1].strip()
                                    data_parts = {p.split(':')[0].strip(): p.split(':')[1].strip() for p in data_line.split('|')}
                                    st.session_state.suggested_ontvanger = data_parts.get("Afzender", "")
                                    st.session_state.suggested_kenmerk = data_parts.get("Kenmerk", "")
                            else:
                                summary_text = full_response_text
                                st.session_state.suggested_action = None

                            st.session_state.current_brief_text = input_text
                            st.session_state.current_summary = summary_text
                            st.session_state.messages = [{"role": "assistant", "content": summary_text}]
                            st.session_state.show_result = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Er ging iets mis tijdens de analyse: {e}")
        else: 
            st.warning("U moet akkoord gaan om door te gaan.")
            
    with tab2:
        # --- Nieuwe layout met zijkolom voor hulp ---
        main_col, help_col = st.columns([2, 1])

        with main_col:
            st.header("Maak een professionele brief")
            
            default_brief_type = "--- Kies een optie ---"
            default_ontvanger = ""
            default_kenmerk = ""
            brief_options = ["--- Kies een optie ---", "Vraag om uitstel van betaling", "Bezwaar maken", "Afspraak afzeggen", "Abonnement opzeggen", "Sollicitatiebrief", "Klacht indienen"]
            action_map = {"Uitstel vragen": "Vraag om uitstel van betaling", "Bezwaar maken": "Bezwaar maken", "Afspraak afzeggen": "Afspraak afzeggen", "Abonnement opzeggen": "Abonnement opzeggen", "Solliciteren": "Sollicitatiebrief", "Klacht indienen": "Klacht indienen"}
            
            if st.session_state.get('prefill_action'):
                suggested_action = st.session_state.get('suggested_action')
                if suggested_action in action_map:
                    default_brief_type = action_map[suggested_action]
                default_ontvanger = st.session_state.get('suggested_ontvanger', "")
                default_kenmerk = st.session_state.get('suggested_kenmerk', "")
                st.session_state.prefill_action = False

            st.write("Kies welk soort brief je wilt sturen en vul de details in.")
            brief_type = st.selectbox("Soort brief", brief_options, index=brief_options.index(default_brief_type))
            toon_keuze = st.selectbox("Toon", ["Zakelijk en formeel", "Vriendelijk maar dringend", "Neutraal en informatief", "Streng en direct", "Zeer boos en ontevreden"])
            
            if brief_type != "--- Kies een optie ---":
                st.write("---")
                st.subheader("Details")
                ontvanger = st.text_input("Aan wie?", value=default_ontvanger)
                kenmerk = st.text_input("Kenmerk (factuurnummer, etc.)?", value=default_kenmerk)
                extra_info = st.text_area("Extra informatie (leg hier je situatie uit):", height=150)
                
                if st.button("Schrijf mijn voorbeeldbrief", type="primary"):
                    if ontvanger:
                        with st.spinner("Ik schrijf een voorbeeldbrief..."):
                            try:
                                schrijf_chain = LLMChain(llm=llm, prompt=PROMPT_SCHRIJVEN_NIEUW)
                                response = schrijf_chain.invoke({
                                    "doel_brief": brief_type,
                                    "ontvanger": ontvanger,
                                    "kenmerk": kenmerk if kenmerk else "Niet van toepassing",
                                    "toon": toon_keuze,
                                    "extra_info": extra_info,
                                    "current_date": datetime.now().strftime("%d-%m-%Y")
                                })
                                brief_tekst = response.get('text', "Kon geen brief genereren.")
                                st.subheader("Jouw voorbeeldbrief:")
                                st.text_area("Je kunt deze tekst kopiëren en aanpassen:", value=brief_tekst, height=500)
                            except Exception as e:
                                st.error(f"Er ging iets mis bij het schrijven van de brief: {e}")
                    else: 
                        st.warning("Vul de naam van de ontvanger in.")

        with help_col:
            st.subheader("💡 Hulp en Tips")
            st.markdown("---")
            with st.expander("**Uitleg per veld**", expanded=True):
                st.markdown("""
                *   **Aan wie?** Vul hier de naam in van de persoon of het bedrijf aan wie je de brief stuurt. Bijvoorbeeld: `Gemeente Amsterdam` of `Mevrouw de Vries`.
                *   **Kenmerk:** Dit is een nummer of code die op de brief staat die je hebt ontvangen. Het helpt de ontvanger om te weten waar jouw brief over gaat. Bijvoorbeeld: `Factuurnummer 12345`.
                *   **Extra informatie:** Vertel hier in je eigen woorden wat je wilt. Wees zo duidelijk mogelijk. Wat is er gebeurd? Wat wil je dat er gebeurt?
                """)

            with st.expander("**Tips voor de inhoud**"):
                 st.markdown("""
                *   **Wees duidelijk:** Schrijf korte zinnen.
                *   **Wees eerlijk:** Vertel precies wat er aan de hand is.
                *   **Vraag om een reactie:** Vraag de ontvanger om te reageren, bijvoorbeeld: "Ik hoor graag binnen twee weken van u."
                *   **Voorbeeld (bij uitstel van betaling):** "Ik kan de rekening nu niet betalen omdat ik mijn baan ben verloren. Kan ik in delen betalen?"
                """)

            with st.expander("**Wat betekent 'Toon'?**"):
                st.markdown("""
                De 'toon' bepaalt hoe de brief klinkt.
                *   **Zakelijk en formeel:** De meest normale keuze voor officiële brieven. Heel netjes.
                *   **Vriendelijk maar dringend:** Als je beleefd wilt blijven, maar het wel belangrijk is dat er snel iets gebeurt.
                *   **Streng en direct:** Als je eerder geen reactie hebt gekregen en duidelijker wilt zijn.
                """)