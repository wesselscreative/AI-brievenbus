import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from PIL import Image
import pytesseract
from gtts import gTTS

# --- Configuratie & Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "Data")

# --- Functies voor de AI-componenten (beter voor caching) ---

@st.cache_resource
def load_vector_db():
    """Laadt de documenten en maakt de vector database aan."""
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", show_progress=True)
    docs = loader.load()
    if not docs:
        st.error(f"Geen documenten gevonden in de '{DATA_PATH}' map. De app kan niet starten.")
        st.stop()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

@st.cache_resource
def get_llm():
    """Initialiseert en retourneert de LLM. Crasht netjes als de key mist."""
    try:
        llm = ChatGroq(
            temperature=0, 
            groq_api_key=st.secrets["GROQ_API_KEY"], 
            model_name="llama3-70b-8192"
        )
        return llm
    except KeyError:
        return None

# --- Prompts ---

PROMPT_UITLEG = PromptTemplate.from_template(
    """
    Je bent een AI-assistent die laaggeletterde mensen in Nederland helpt met het begrijpen van officiële brieven.
    Jouw taak is om de belangrijkste punten uit een brief te halen. Gebruik de CONTEXT om je antwoord te baseren op vergelijkbare documenten.

    ANALYSEER EERST DE VRAAG. Als de tekst in de VRAAG duidelijk GEEN officiële brief is (bijvoorbeeld een recept, een nieuwsartikel, of onzin), negeer dan de CONTEXT en antwoord letterlijk en alleen met de volgende zin:
    "Deze tekst lijkt geen officiële brief te zijn. Ik kan je hier helaas niet mee helpen. Probeer een foto te maken van een brief van bijvoorbeeld de overheid, een bedrijf of een ziekenhuis."

    Als de tekst in de VRAAG WEL een officiële brief lijkt, vat dan de belangrijkste punten samen in simpele taal (B1-niveau). Focus op: Wie heeft de brief gestuurd? Wat moet de persoon DOEN? Staat er een bedrag of datum in?

    CONTEXT: {context}
    VRAAG: {question}
    HELPENDE ANTWOORD IN SIMPELE TAAL:
    """
)

# --- Hoofdapplicatie ---

st.set_page_config(page_title="Hulp bij Moeilijke Brieven", layout="wide")
st.title("🤖 AI Hulp voor Moeilijke Brieven")

# Initialiseer de AI en toon een foutmelding als het misgaat
llm = get_llm()
if not llm:
    st.error("GROQ API sleutel niet gevonden. Zorg dat je deze hebt ingesteld in de Streamlit Cloud secrets.")
    st.stop()

# --- Tabbladen ---
tab1, tab2 = st.tabs(["✉️ **Brief Laten Uitleggen**", "✍️ **Help Mij Schrijven**"])


# --- Tab 1: Brief Uitleggen ---
with tab1:
    st.header("Laat je brief uitleggen")
    st.write("Plak de tekst van een moeilijke brief hieronder, of maak een foto van je brief.")
    
    agreed_tab1 = st.checkbox("Ik begrijp dat mijn brief anoniem verwerkt wordt en ga akkoord.", key="agree_tab1")

    if agreed_tab1:
        # Bouw de RAG-keten specifiek voor deze tab
        vectorstore = load_vector_db()
        retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)
        combine_docs_chain = create_stuff_documents_chain(llm, PROMPT_UITLEG)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

        # UI elementen
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Optie 1: Maak een foto")
            uploaded_file = st.file_uploader("Kies een afbeelding...", type=["jpg", "png", "jpeg"], key="uploader_tab1")
            if uploaded_file:
                with st.spinner('Bezig met het lezen van de foto...'):
                    image = Image.open(uploaded_file)
                    text = pytesseract.image_to_string(image, lang='nld')
                    st.session_state.text_from_ocr = text # Gebruik session_state om tekst te bewaren
                    st.success("Tekst uit foto gehaald!")
        
        with col2:
            st.subheader("Optie 2: Plak de tekst")
            user_input = st.text_area("Tekst van de brief:", value=st.session_state.get('text_from_ocr', ''), height=300, key="input_tab1")

        if st.button("Leg de brief uit", type="primary", key="button_tab1"):
            if user_input:
                with st.spinner("Ik lees de brief en maak een simpele samenvatting..."):
                    full_question = f"Vat de belangrijkste punten van deze brief samen: '{user_input}'"
                    try:
                        response = rag_chain.invoke({"question": full_question})
                        
                        # Veilige controle of het antwoord bestaat
                        if response and 'answer' in response and response['answer']:
                            answer = response['answer']
                            st.subheader("Simpele Uitleg:")
                            st.markdown(f"<div style='border-left: 5px solid #007bff; padding: 10px; background-color: #f0f2f6; border-radius: 5px;'>{answer}</div>", unsafe_allow_html=True)
                            
                            st.subheader("Lees de uitleg voor:")
                            tts = gTTS(text=answer, lang='nl', slow=False)
                            tts.save("uitleg.mp3")
                            st.audio("uitleg.mp3")
                        else:
                            st.error("Het is niet gelukt om een duidelijke samenvatting te maken. Dit kan gebeuren als de tekst geen duidelijke brief is. Probeer het opnieuw.")
                    
                    except Exception as e:
                        st.error(f"Er is een onverwachte technische fout opgetreden. Probeer het later opnieuw. Details: {e}")
            else:
                st.warning("Plak eerst tekst of upload een foto van een brief.")


# --- Tab 2: Schrijfhulp ---
with tab2:
    st.header("Maak een professionele brief in simpele stappen")
    st.write("Kies welk soort brief je wilt sturen en vul de details in. De AI schrijft dan een voorbeeld voor je.")

    brief_type = st.selectbox(
        "**Stap 1: Kies wat voor soort brief je wilt schrijven**",
        [
            "--- Kies een optie ---", "Vraag om uitstel van betaling", "Bezwaar maken tegen een boete of beslissing",
            "Een afspraak afzeggen of verzetten", "Een abonnement of contract opzeggen"
        ],
        key="brief_type_select"
    )

    if brief_type != "--- Kies een optie ---":
        st.write("---")
        st.subheader("**Stap 2: Vul de details in**")
        ontvanger = st.text_input("Aan wie is de brief? (Naam bedrijf of persoon)", key="ontvanger_input")
        kenmerk = st.text_input("Wat is het kenmerk, factuurnummer of klantnummer?", key="kenmerk_input")
        
        extra_info_prompt = ""
        # Dynamische prompts voor extra info
        if "uitstel" in brief_type: extra_info_prompt = "Waarom vraag je uitstel en wanneer kun je wel betalen? (Schrijf dit in je eigen woorden)"
        elif "Bezwaar" in brief_type: extra_info_prompt = "Waarom ben je het niet eens met de boete of beslissing? (Schrijf dit in je eigen woorden)"
        elif "afspraak" in brief_type: extra_info_prompt = "Over welke afspraak gaat het (datum en tijd)? En wanneer zou je eventueel een nieuwe afspraak willen?"
        elif "abonnement" in brief_type: extra_info_prompt = "Per wanneer wil je opzeggen? (Optioneel)"
        extra_info = st.text_area(extra_info_prompt, key=f"extra_info_{brief_type}")

        st.write("---")
        st.subheader("**Stap 3: Maak de brief**")
        if st.button("Schrijf mijn voorbeeldbrief", type="primary", key="button_tab2"):
            if ontvanger and kenmerk:
                schrijf_prompt_template = f"""
                Schrijf een korte, beleefde en formele Nederlandse brief namens een persoon. Gebruik simpele en duidelijke taal (B1-niveau).
                Het doel van de brief is: '{brief_type}'. De brief is gericht aan: {ontvanger}.
                Het relevante kenmerk/nummer is: {kenmerk}. Aanvullende informatie van de gebruiker: '{extra_info}'.
                Zorg voor een correcte aanhef, een duidelijke kernboodschap en een formele afsluiting met ruimte voor een naam en handtekening. Houd het kort en krachtig.
                """
                
                with st.spinner("Ik schrijf een voorbeeldbrief voor je..."):
                    try:
                        response = llm.invoke(schrijf_prompt_template)
                        st.subheader("Jouw voorbeeldbrief:")
                        st.text_area("Je kunt deze tekst kopiëren en gebruiken:", value=response.content, height=400, key="result_brief")
                    except Exception as e:
                        st.error(f"Er is iets misgegaan bij het schrijven van de brief. Fout: {e}")
            else:
                st.warning("Vul alsjeblieft de naam van de ontvanger en het kenmerk in.")


# --- Privacyverklaring onderaan ---
st.markdown("---")
with st.expander("Privacy en Veiligheid (Lees dit)"):
    st.write("""
    **Jouw privacy is voor ons het allerbelangrijkste.**
    
    - **Wij slaan jouw brieven en foto's NIET op.** Alles wat je plakt of uploadt, wordt direct na gebruik verwijderd uit het geheugen van deze website.
    - Om de uitleg te kunnen geven, sturen we de tekst van je brief **anoniem** naar een beveiligde AI-dienst (Groq, die het Llama3-model draait).
    - Deze dienst heeft een strikt privacybeleid en gebruikt jouw gegevens **niet** om hun AI te trainen. Na een korte periode wordt je data ook daar permanent verwijderd.
    - Wij gebruiken geen tracking- of marketingcookies.
    """)