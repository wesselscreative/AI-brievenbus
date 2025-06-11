import pandas as pd
import os
import time
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# --- Configuratie ---
# Deze instellingen zorgen ervoor dat het script de juiste bestanden en mappen vindt.
DATA_DIR = "Data"
METADATA_FILE = "metadata.csv"
LLM_MODEL = "llama3-8b-8192"  # We gebruiken het snelle 8b model, perfect voor deze taak

# --- Voorbereiding: Laad de API Key ---
# Dit laadt de variabelen uit je .env bestand.
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Een duidelijke foutmelding als de API key niet gevonden kan worden.
if not groq_api_key:
    raise ValueError("GROQ_API_KEY niet gevonden. Zorg dat je een .env bestand hebt met de key.")

# --- Initialiseer de LLM ---
# We maken één keer verbinding met de Groq AI service.
llm = ChatGroq(model_name=LLM_MODEL, groq_api_key=groq_api_key)

# --- De Prompt voor de Samenvatting ---
# Dit is de precieze instructie die we aan de AI geven voor elke brief.
prompt_template_str = """
Je bent een expert in het schrijven van teksten op A2-taalniveau voor laaggeletterden.
Vat de volgende officiële brief samen in 2 tot 4 zeer simpele, korte zinnen.
Focus alleen op de allerbelangrijkste actie of boodschap.
Gebruik geen moeilijke woorden. Het resultaat moet direct en duidelijk zijn.

BRIEFTEKST:
"{brieftekst}"

A2-SAMENVATTING:
"""
prompt = PromptTemplate.from_template(prompt_template_str)


# --- Hoofdfunctie ---
def genereer_samenvattingen():
    """
    Leest de metadata.csv, genereert voor elke lege samenvatting een nieuwe,
    en slaat het complete bestand weer op.
    """
    try:
        df = pd.read_csv(METADATA_FILE)
    except FileNotFoundError:
        print(f"FOUT: '{METADATA_FILE}' niet gevonden in de projectmap. Zorg dat het bestand bestaat.")
        return

    # We maken een kopie om veilig aanpassingen te doen tijdens het doorlopen van de lijst.
    df_to_update = df.copy()

    # Loop door elke rij in het metadata bestand.
    for index, row in df.iterrows():
        # Sla over als er al een samenvatting is (en het is geen lege placeholder).
        if pd.notna(row['a2_samenvatting']) and row['a2_samenvatting'] != '[leeg]':
            print(f"Skipping {row['bestandsnaam']} (heeft al een samenvatting).")
            continue

        filepath = os.path.join(DATA_DIR, row['bestandsnaam'])
        
        # Dit blok vangt alle mogelijke fouten voor één specifiek bestand op,
        # zodat het script niet stopt als er één bestand problemen geeft.
        try:
            content = None
            # Poging 1: Probeer het bestand te openen met de standaard UTF-8 encoding.
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Poging 2: Als UTF-8 mislukt, probeer dan een veelvoorkomende Windows-encoding.
                print(f"INFO: UTF-8 mislukt voor {row['bestandsnaam']}, probeer 'cp1252'...")
                with open(filepath, 'r', encoding='cp1252') as f:
                    content = f.read()

            if content is None:
                raise ValueError("Kon het bestand niet lezen met de geprobeerde encodings.")

            # Creëer de 'keten' van taken: stop de brieftekst in de prompt, en stuur naar de AI.
            chain = prompt | llm
            
            # Voer de taak uit.
            summary_obj = chain.invoke({"brieftekst": content})
            summary = summary_obj.content.strip()  # .content is nodig voor Chat-modellen
            
            # Sla de gegenereerde samenvatting op in onze data.
            df_to_update.loc[index, 'a2_samenvatting'] = summary
            print(f"({index + 1}/{len(df)}) Samenvatting gemaakt voor: {row['bestandsnaam']}")
            
            # Wacht een klein moment om de API niet te overbelasten.
            time.sleep(0.5)
            
        except FileNotFoundError:
            print(f"WAARSCHUWING: Bestand niet gevonden: {filepath}. Wordt overgeslagen.")
        except Exception as e:
            print(f"FOUT bij het verwerken van {row['bestandsnaam']}: {e}")

    # Sla het volledig bijgewerkte bestand op.
    df_to_update.to_csv(METADATA_FILE, index=False, encoding='utf-8')
    print(f"\nAlle samenvattingen zijn gegenereerd en opgeslagen in '{METADATA_FILE}'!")


# --- Start het Script ---
# Deze regel zorgt ervoor dat de functie 'genereer_samenvattingen' wordt uitgevoerd
# wanneer je 'python genereer_samenvattingen.py' in de terminal typt.
if __name__ == "__main__":
    genereer_samenvattingen()