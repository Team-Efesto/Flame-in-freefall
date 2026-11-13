import os
import glob
import pandas as pd
import json

def process_nasa_folder(folder_path, output_json_path, require_target=None):
    print(f"📂 Avvio scansione nella cartella: {folder_path}")
    
    files = glob.glob(os.path.join(folder_path, "*.csv")) + glob.glob(os.path.join(folder_path, "*.xlsx"))
    
    if not files:
        print("❌ Nessun file CSV o XLSX trovato nella cartella specificata.")
        return

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
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, low_memory=False)
            else:
                df = pd.read_excel(file_path)
            
            df = df.rename(columns=column_mapping)
            
            if require_target and require_target in df.columns:
                df = df.dropna(subset=[require_target])
            
            # Conversione sicura in numeri senza errori sulle colonne testuali
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
            
            # Conversione garantita dei valori mancanti in None (diventa 'null' nel JSON)
            df = df.astype(object).where(pd.notnull(df), None)
            df['source_file'] = file_name
            
            records = df.to_dict(orient='records')
            all_records.extend(records)
            total_files_processed += 1
            
        except Exception as e:
            print(f"⚠️ Errore nel file {file_name}: {e}")

    print(f"✅ Letti {total_files_processed} file. Totale sample estratti: {len(all_records)}")

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    ml_dataset = {
        "dataset_metadata": {
            "source": "NASA Physical Science Informatics (PSI)",
            "processing_mode": "Dynamic Feature Extraction",
            "total_files_merged": total_files_processed,
            "total_samples": len(all_records),
            "features_description": "Columns dynamically merged. Missing values mapped to 'null'."
        },
        "training_data": all_records
    }

    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(ml_dataset, json_file, indent=4, ensure_ascii=False)
    print(f"🚀 JSON salvato in: {output_json_path}")

if __name__ == "__main__":
    CARTELLA_DATI = r"./dati_nasa_raw/tabelle"
    FILE_JSON_OUTPUT = r"./data/neural_network_dataset.json"
    COLONNA_TARGET = None  # Imposta "flame_spread_rate_mm_s" se vuoi filtrare solo le righe con quel target

    process_nasa_folder(CARTELLA_DATI, FILE_JSON_OUTPUT, require_target=COLONNA_TARGET)