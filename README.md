read.md
# AI Hulp voor Moeilijke Brieven (Prototype)

## 🧠 Beschrijving
Een digitale assistent die complexe, officiële brieven vertaalt naar begrijpelijke taal (A2-niveau). 
Dit project gebruikt de kracht van Large Language Models om een brug te slaan over taalbarrières en mensen meer controle te geven over hun eigen administratie.

## 🎯 Het Probleem 
Officiële documenten van overheden, verzekeraars of incassobureaus zijn vaak geschreven in complexe, formele taal. Voor miljoenen mensen in Nederland die moeite hebben met lezen, leidt dit tot stress, misverstanden en gemiste deadlines met potentieel grote gevolgen.

## 💡 De Oplossing 
Deze tool analyseert de inhoud van een brief en zet deze om in een eenvoudige, gestructureerde samenvatting. Het beantwoordt de belangrijkste vragen:
- Van wie is deze brief?
- Wat moet ik doen?
- Staan er bedragen of data in die ik moet weten?

Daarnaast helpt de tool de gebruiker bij het opstellen van een passende reactie.

## 🚀 Live Demo
➡️ Check hier de tool om live te proberen! --> https://ai-brievenhulp.streamlit.app/

## ✨ Belangrijkste Features
- 📄 Simpele Samenvattingen: Upload een PDF, afbeelding (JPG/PNG) of plak tekst en ontvang direct een duidelijke uitleg op A2-niveau.
- ✍️ Hulp bij Schrijven: Genereer een professioneel concept voor een formele brief (B1-niveau) voor o.a. uitstel van betaling, bezwaar of een klacht.
- 🔊 Voorleesfunctie: Laat de samenvatting voorlezen met een duidelijke, rustige stem.
- 🌐 Vertaalopties: Vertaal de samenvatting naar het Engels, Arabisch, Turks of Pools.
- 💬 Interactieve Chat: Stel vervolgvragen over de brief om specifieke details op te helderen.
- 🕵️ Actie-suggesties: De tool identificeert de meest logische vervolgstap en kan het formulier voor het schrijven van een brief alvast voor je invullen.

## 📦 Installatie
```bash
python -m venv venv
source venv/bin/activate  # of venv\Scripts\activate op Windows
pip install -r requirements.txt
