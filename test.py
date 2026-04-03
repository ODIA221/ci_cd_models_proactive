# Créez un fichier test.py
cat > test.py << 'EOF'
import pandas as pd
import numpy as np
from pathlib import Path

# Cherchez vos fichiers CSV
data_path = Path('data/raw')
csv_files = list(data_path.glob('*.csv'))

if csv_files:
    df = pd.read_csv(csv_files[0])
    print(f"✅ Fichier chargé: {csv_files[0].name}")
    print(f"📊 Dimensions: {df.shape}")
    print(f"📋 Colonnes: {df.columns.tolist()[:5]}...")
else:
    print("❌ Aucun fichier CSV trouvé dans data/raw/")
    print("   Placez vos données dans ce dossier")
EOF

# Exécutez le script
#python test.py