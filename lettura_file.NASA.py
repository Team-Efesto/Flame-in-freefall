import os
import glob
import pandas as pd
import numpy as np
import json

def process_nasa_folder(folder_path, output_json_path, require_target=None):
    """
    Scansiona una cartella locale scaricata, legge tutti i CSV/Excel degli esperimenti, 
    standardizza le colonne principali, mantiene TUTTE le colonne aggiuntive e salva un JSON unico.
    """
    print(f"📂 Avvio scansione nella cartella: {folder_path}")
    
    # Trova tutti i file CSV ed Excel nella cartella (ignora le sottocartelle)
    files = glob.glob(os.path.join(folder_path, "*.csv")) + glob.glob(os.path.join(folder_path, "*.xlsx"))
    
    if not files:
        print("❌ Nessun file dati trovato nella cartella specificata.")
        return

    # Mappatura delle sole colonne di base. 
    # Le altre NON presenti in questo dizionario verranno tenute così come sono.
    column_mapping = {
        'Experiment_ID': 'experiment_id',
        'Fuel_Material': 'fuel_type',
        'O2_Concentration_pct': 'o2_percentage',
        'Pressure_psia': 'pressure_psi',
        'Airflow_Velocity_cms': 'airflow_velocity_cm_s',
        'Flame_Spread_Rate_mms': 'flame_spread_rate_mm_s'
    }

    all_records = []
    total_files_processed = 0

    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f"🔄 Elaborazione file: {file_name}")
        
        try:
            # 1. Lettura dinamica del file
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, low_memory=False)
            else:
                df = pd.read_excel(file_path)
            
            # 2. Rinomina solo le colonne standard che riesce a trovare (ignorando i missing)
            df = df.rename(columns=column_mapping)
            
            # 3. Drop condizionale (Opzionale)
            # Se stiamo addestrando una rete e ci serve un target (es. flame_spread_rate_mm_s),
            # scartiamo le righe in cui questo target manca. Altrimenti teniamo tutto.
            if require_target and require_target in df.columns:
                df = df.dropna(subset=[require_target])
            
            # 4. Conversione Smart dei dati per TUTTE le colonne
            for col in df.columns:
                # Prova a convertire in numero. Se fallisce (perché è una vera stringa testuale),
                # ignora l'errore e lascia il testo. Se è un numero vuoto, diventa NaN.
                df[col] = pd.to_numeric(df[col], errors='ignore')
            
            # 5. Gestione standardizzata dei valori mancanti -> Nessuna assunzione
            # Converte np.nan, pandas.NA, ecc. in None (che diventerà 'null' nel JSON)
            df = df.replace({np.nan: None, float('nan'): None, pd.NA: None})
            
            # Aggiunge una colonna di tracciamento per sapere da quale file arriva il dato
            df['source_file'] = file_name
            
            # Trasforma le righe del dataframe in dizionari e uniscili alla lista globale
            records = df.to_dict(orient='records')
            all_records.extend(records)
            total_files_processed += 1
            
        except Exception as e:
            print(f"⚠️ Errore saltato nel file {file_name}: {e}")

    print(f"✅ Letti {total_files_processed} file. Totale sample estratti: {len(all_records)}")

    # Strutturazione per il file JSON finale
    ml_dataset = {
        "dataset_metadata": {
            "source": "NASA Physical Science Informatics (PSI)",
            "processing_mode": "Dynamic Feature Extraction",
            "total_files_merged": total_files_processed,
            "total_samples": len(all_records),
            "features_description": "Columns dynamically merged. Missing values mapped to 'null' for masking layers."
        },
        "training_data": all_records
    }

    # Salvataggio su disco
    try:
        with open(output_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(ml_dataset, json_file, indent=4, ensure_ascii=False)
        print(f"🚀 Database JSON per la rete neurale salvato in: {output_json_path}")
    except Exception as e:
        print(f"❌ Errore durante il salvataggio del JSON: {e}")

# ==========================================
# ESECUZIONE
# ==========================================
if __name__ == "__main__":
    # Inserisci il percorso della cartella che hai scaricato sul tuo PC
    # Esempio per Windows: r"C:\Users\tuonome\Downloads\Nasa_Datasets"
    CARTELLA_DOWNLOAD = r"./dati_nasa_raw" 
    
    # Nome del file JSON di output
    FILE_JSON_OUTPUT = "neural_network_dataset.json"
    
    # Se vuoi scartare le righe in cui manca il dato che la rete deve prevedere, 
    # inserisci il nome della colonna. Se vuoi tenere tutti i dati, scrivi None.
    COLONNA_TARGET = "flame_spread_rate_mm_s" # oppure None

    process_nasa_folder(CARTELLA_DOWNLOAD, FILE_JSON_OUTPUT, require_target=COLONNA_TARGET)