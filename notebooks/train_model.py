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

