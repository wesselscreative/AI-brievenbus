import streamlit as st
import os
from langchain_community.llms import Ollama
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.multi_query import MultiQueryRetriever

# Importeer de nieuwe bibliotheken voor OCR en TTS
from PIL import Image
import pytesseract
from gtts import gTTS

# --- Configuratie & Setup (Robuuste versie) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "Data")
LLM_MODEL = "llama3"
EMBEDDING_MODEL = "llama3"

# --- Functies ---

@st.cache_resource
def load_or_create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", show_progress=True)
    docs = loader.load()
    if not docs:
        st.error(f"Geen documenten gevonden in de '{DATA_PATH}' map.")
        st.stop()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

prompt_template_string = """
Je bent een AI-assistent die laaggeletterde mensen in Nederland helpt met het begrijpen van moeilijke brieven.
Gebruik de volgende stukken opgehaalde context om de vraag van de gebruiker te beantwoorden.
- Antwoord altijd in het Nederlands.
- Gebruik extreem simpele taal (B1-niveau of lager). Gebruik korte zinnen.
- Geef geen mening en verzin geen informatie die niet in de context staat.
- Focus op de allerbelangrijkste informatie:
  1. Wie heeft de brief gestuurd?
  2. Wat moet de persoon DOEN? (bijv. betalen, bellen, iets opsturen, naar een afspraak gaan).
  3. Staat er een bedrag in? Zo ja, hoeveel? Moet de persoon betalen of krijgt hij/zij geld?
  4. Staat er een deadline of datum in? Zo ja, welke?
- Begin je antwoord altijd met een duidelijke samenvatting in één zin.

CONTEXT:
{context}

VRAAG:
{question}

HELPENDE ANTWOORD IN SIMPELE TAAL:
"""
prompt = PromptTemplate.from_template(prompt_template_string)

# --- Hoofdapplicatie ---

st.set_page_config(page_title="Hulp bij Moeilijke Brieven", layout="wide")
st.title("🤖 AI Hulp voor Moeilijke Brieven")
st.write("Plak de tekst van een moeilijke brief hieronder, of maak een foto van je brief.")

# --- Bouw de RAG-keten (gebeurt maar 1 keer) ---
with st.spinner("De AI-assistent wordt klaargezet. Dit kan even duren..."):
    vectorstore = load_or_create_vector_db()
    llm = Ollama(model=LLM_MODEL)
    # Gebruik de slimmere retriever
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(), llm=llm
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever_from_llm, question_answer_chain)

# --- Gebruikersinterface met OCR en TTS ---

# Initialiseer session state om tekst tussen runs te bewaren
if 'text_from_ocr' not in st.session_state:
    st.session_state.text_from_ocr = ""

# Kolommen voor de invoer
col1, col2 = st.columns(2)

with col1:
    st.subheader("Optie 1: Maak een foto van je brief")
    uploaded_file = st.file_uploader("Kies een afbeelding...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        with st.spinner('Bezig met het lezen van de foto...'):
            try:
                image = Image.open(uploaded_file)
                # Gebruik 'nld' voor Nederlandse taal
                text = pytesseract.image_to_string(image, lang='nld')
                st.session_state.text_from_ocr = text
                st.success("Tekst uit foto gehaald!")
            except Exception as e:
                st.error(f"Fout bij het lezen van de afbeelding. Zorg dat Tesseract correct is geïnstalleerd. Fout: {e}")

with col2:
    st.subheader("Optie 2: Plak de tekst van je brief")
    user_input = st.text_area("Tekst van de brief:", value=st.session_state.text_from_ocr, height=300)

if st.button("Leg de brief uit", type="primary"):
    # Gebruik de tekst uit het tekstvak, die mogelijk gevuld is door OCR
    final_input = user_input
    if final_input:
        with st.spinner("Ik lees de brief en maak een simpele samenvatting..."):
            full_question = f"Vat de belangrijkste punten van deze brief samen: '{final_input}'"
            try:
                response = rag_chain.invoke({"question": full_question})
                answer = response['answer']
                
                st.subheader("Simpele Uitleg:")
                st.markdown(f"<div style='border-left: 5px solid #007bff; padding: 10px; background-color: #f0f2f6; border-radius: 5px;'>{answer}</div>", unsafe_allow_html=True)
                
                # --- TTS SECTIE ---
                st.subheader("Lees de uitleg voor:")
                try:
                    tts = gTTS(text=answer, lang='nl', slow=False)
                    tts.save("uitleg.mp3")
                    st.audio("uitleg.mp3")
                except Exception as e:
                    st.error(f"Kon de audio niet aanmaken. Fout: {e}")
                # --- EINDE TTS SECTIE ---

                with st.expander("Bekijk welke informatie is gebruikt"):
                    for i, doc in enumerate(response["context"]):
                        source_file = os.path.basename(doc.metadata.get('source', 'Onbekend'))
                        st.info(f"**Bron {i+1}: `{source_file}`**\n\n*Fragment:*\n...{doc.page_content[50:250]}...")

            except Exception as e:
                st.error(f"Er is iets misgegaan. Zorg ervoor dat Ollama draait. Foutmelding: {e}")
    else:
        st.warning("Plak eerst tekst of upload een foto van een brief.")

    st.markdown("---") # Dit voegt een scheidingslijn toe
with st.expander("Privacy en Veiligheid (Lees dit)"):
    st.write("""
    **Jouw privacy is voor ons het allerbelangrijkste.**

    *   **Wij slaan jouw brieven en foto's NIET op.** Alles wat je plakt of uploadt, wordt alleen tijdelijk gebruikt om je de uitleg te geven.
    *   Zodra je de pagina vernieuwt of sluit, wordt alle informatie **direct en permanent verwijderd**.
    *   Er wordt niets naar ons of naar anderen gestuurd. De analyse gebeurt binnen deze beveiligde omgeving.
    *   Deze website gebruikt alleen de noodzakelijke functionele cookies om de app te laten werken. Wij gebruiken geen tracking- of marketingcookies.

    Je kunt deze tool veilig gebruiken.
    """)