import pandas as pd # type: ignore
import numpy as np # type: ignore
from pathlib import Path

# Cherchez vos fichiers CSV
data_path = Path('./data/raw/metrics/dataset_metrics')
csv_files = list(data_path.glob('*.csv'))

if csv_files:
    df = pd.read_csv(csv_files[0])
    print(f"✅ Fichier chargé: {csv_files[0].name}")
    print(f"📊 Dimensions: {df.shape}")
    print(f"📋 Colonnes: {df.columns.tolist()[:5]}...")
    print(f"\n🔍 Aperçu:")
    print(df.head())
else:
    print("❌ Aucun fichier CSV trouvé dans data/raw/metrics/")
    print("   Placez vos données dans ce dossier")