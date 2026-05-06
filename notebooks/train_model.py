import pandas as pd
import numpy as np
import warnings


warnings.filterwarnings("ignore", category=UserWarning)

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

from sklearn.ensemble import RandomForestClassifier

# Créer le modèle
model = RandomForestClassifier(
    n_estimators=100,  # Le modèle va construire 100 arbres de décision
    random_state=42    # Pour garantir la reproductibilité des résultats
)

# Entraîner le modèle sur les données d'entraînement
model.fit(X_train, y_train)

print("Modèle entraîné !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")

# Prédire les diagnostics sur les données de test
y_pred = model.predict(X_test)

# Créer un tableau pour comparer les 10 premières prédictions avec les vraies valeurs
comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prédiction': y_pred[:10]
})

# Afficher la comparaison
print("Comparaison des 10 premiers résultats :")
print(comparison)

from sklearn.metrics import accuracy_score

# Calcul de la précision (pourcentage de bonnes prédictions)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.2%}")

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Matrice de confusion (calcul)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# 2. Rapport de classification détaillé
print("Rapport de classification :")
print(classification_report(y_test, y_pred))

import joblib
import os

target_dir = os.path.join("..", "models")
target_file = os.path.join(target_dir, "model.pkl")

# Créer le dossier s'il n'existe pas
os.makedirs(target_dir, exist_ok=True)

# Sérialiser le modèle
joblib.dump(model, target_file)

# Vérifier la taille
size = os.path.getsize(target_file)
print(f"Modèle sauvegardé : {target_file}")
print(f"Taille : {size / 1024:.1f} Ko")

import joblib
import os

# Sauvegarder les encodeurs (indispensables pour les nouvelles données)
# On utilise ".." pour remonter à la racine depuis le dossier 'notebooks'
joblib.dump(le_sexe, "../models/encoder_sexe.pkl")
joblib.dump(le_region, "../models/encoder_region.pkl")

# Sauvegarder la liste des features (pour référence lors de l'inférence)
joblib.dump(feature_cols, "../models/feature_cols.pkl")

print("Encodeurs et metadata sauvegardés dans le dossier models.")

# Simuler ce que fera l'API :
# Charger le modèle DEPUIS LE FICHIER (pas depuis la mémoire)
model_loaded = joblib.load("../models/model.pkl")
le_sexe_loaded = joblib.load("../models/encoder_sexe.pkl")
le_region_loaded = joblib.load("../models/encoder_region.pkl")

print(f"Modèle rechargé : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")

# Un nouveau patient arrive au centre de santé de Médina
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'region': 'Dakar'
}

# Encoder les valeurs catégoriques
sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

# Préparer le vecteur de features
features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

# Prédire
diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]
proba_max = probas.max()

print(f"\n--- Résultat du pré-diagnostic ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilité : {proba_max:.1%}")

print(f"\nProbabilités par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"{classe:8s} : {proba:.1%} {bar}")


# Affichons l'importance des features pour comprendre les facteurs clés du diagnostic
importances = model.feature_importances_
for name, imp in sorted(zip(feature_cols, importances),
                        key=lambda x: x[1], reverse=True):
    print(f"  {name:20s} : {imp:.3f}")


#Testons avec plusieurs patients fictifs pour voir les diagnostics et les probabilités associées
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(BASE_DIR, "models")

try:
    model_loaded = joblib.load(os.path.join(MODELS_PATH, "model.pkl"))
    le_sexe_loaded = joblib.load(os.path.join(MODELS_PATH, "encoder_sexe.pkl"))
    le_region_loaded = joblib.load(os.path.join(MODELS_PATH, "encoder_region.pkl"))
    print("Modèles et encodeurs chargés avec succès.\n")
except FileNotFoundError:
    print("Erreur : Le dossier 'models' ou les fichiers .pkl sont introuvables au chemin attendu.")
    exit()

# 2. Liste des patients (Données propres)
patients = [
    {
        "nom": "Patient 1 - Jeune sans symptômes",
        "age": 20, "sexe": "M", "temperature": 37.0,
        "tension_sys": 120, "toux": False, "fatigue": False,
        "maux_tete": False, "region": "Dakar"
    },
    {
        "nom": "Patient 2 - Adulte avec forte fièvre",
        "age": 35, "sexe": "F", "temperature": 40.2,
        "tension_sys": 95, "toux": False, "fatigue": True,
        "maux_tete": True, "region": "Thiès"
    },
    {
        "nom": "Patient 3 - Patient âgé avec toux",
        "age": 68, "sexe": "M", "temperature": 38.5,
        "tension_sys": 140, "toux": True, "fatigue": True,
        "maux_tete": False, "region": "Saint-Louis"
    }
]

# 3. Boucle de prédiction avec gestion d'erreurs d'encodage
print("Résultats du pré-diagnostic")
for p in patients:
    try:
        # Transformation des données textuelles en nombres
        # .strip() retire les espaces accidentels en début/fin de chaîne
        sexe_enc = le_sexe_loaded.transform([p["sexe"].strip()])[0]
        region_enc = le_region_loaded.transform([p["region"].strip()])[0]

        # Préparation du vecteur de caractéristiques (Features)
        # L'ordre doit être identique à celui de l'entraînement (X_train)
        features = [
            p["age"], 
            sexe_enc, 
            p["temperature"], 
            p["tension_sys"],
            int(p["toux"]), 
            int(p["fatigue"]), 
            int(p["maux_tete"]), 
            region_enc
        ]

        # Calcul des prédictions
        diagnostic = model_loaded.predict([features])[0]
        probas = model_loaded.predict_proba([features])[0]
        proba_max = probas.max()

        # Affichage propre
        print(f"\n{p['nom']}")
        print(f"Diagnostic : {diagnostic}")
        print(f"Confiance  : {proba_max:.1%}")

    except ValueError as e:
        print(f"\n Erreur pour {p['nom']} : {e}")
        print(" La région ou le sexe n'est pas reconnu par le modèle.")