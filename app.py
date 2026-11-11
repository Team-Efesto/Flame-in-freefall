import streamlit as st
import pandas as pd

# Titolo principale dell'applicazione
st.title("🔥 Flame in Freefall - Prototipo Dashboard")

# Un semplice testo o sottotitolo
st.write("Benvenuto nel pannello di controllo degli esperimenti NASA.")

# Inserire un componente interattivo (Sidebar con filtri)
st.sidebar.header("Filtri di Ricerca")
livello_ossigeno = st.sidebar.slider("Concentrazione di O2 (%)", 21, 100, 30)

# Mostrare il valore selezionato
st.subheader(f"Analisi per il livello di O2: {livello_ossigeno}%")

# Esempio di tabella dati
data = {
    "Esperimento": ["BASS-II", "Saffire-I", "ACME CFI-G"],
    "Combustibile": ["Solido (Plastica)", "Solido (Tessuto)", "Gas (Propano)"],
    "O2 (%)": [21, 23, 85]
}
df = pd.DataFrame(data)

# Visualizzazione della tabella filtrata
st.dataframe(df[df["O2 (%)"] <= livello_ossigeno])

#ciao