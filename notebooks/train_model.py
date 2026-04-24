import pandas as pd
import numpy as np

# Charger le dataset
file_path = "../data/patients_dakar.csv"
df = pd.read_csv(file_path)

# Vérifier les dimensions
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")

# Afficher les informations structurelles
print(f"\nColonnes : {list(df.columns)}")

# Distribution des diagnostics
print("\nDiagnostics :")
print(df['diagnostic'].value_counts())

from sklearn.preprocessing import LabelEncoder

# Encoder les variables catégoriques en nombres
# Rappel : les modèles de Machine Learning ne traitent que des données numériques !
le_sexe = LabelEncoder()
le_region = LabelEncoder()

# Transformation des colonnes 'sexe' et 'region'
df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Définir les variables explicatives (X) et la variable cible (y)
feature_cols = [
    'age', 
    'sexe_encoded', 
    'temperature', 
    'tension_sys',
    'toux', 
    'fatigue', 
    'maux_tete', 
    'region_encoded'
]

X = df[feature_cols]
y = df['diagnostic']

# Affichage des dimensions pour vérification
print(f"Features : {X.shape}")  # Exemple : (500, 8)
print(f"Cible    : {y.shape}")    # Exemple : (500,)

from sklearn.model_selection import train_test_split

# Division du dataset : 80% pour l'entraînement, 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y,
    test_size=0.2,    # 20% des données réservées au test
    random_state=42,  # Garantit la reproductibilité des résultats
    stratify=y        # Maintient les proportions des classes (diagnostics)
)

print(f"Entraînement : {X_train.shape[0]} patients")
print(f"Test         : {X_test.shape[0]} patients")