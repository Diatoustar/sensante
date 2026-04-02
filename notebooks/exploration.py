"""
SenSante - Exploration du dataset patients_dakar.csv
Lab 1 : Git, Python et Structure Projet
"""

import pandas as pd

# ===== CHARGER LES DONNEES =====
# Assure-toi que le fichier est bien dans le dossier 'data'
df = pd.read_csv("data/patients_dakar.csv")

# ===== ENTÊTE =====
print("=" * 50)
print("SENSANTE - Exploration du dataset")
print("=" * 50)

# Dimensions du dataset
print(f"\nNombre de patients : {len(df)}")
print(f"Nombre de colonnes : {df.shape[1]}")
print(f"Colonnes : {list(df.columns)}")

# Aperçu des 5 premières lignes
print(f"\n--- 5 premiers patients ---")
print(df.head())

# ===== REPARTITION DES DIAGNOSTICS =====
print(f"\n--- Repartition des diagnostics ---")
diag_counts = df["diagnostic"].value_counts()
for diag, count in diag_counts.items():
    pct = (count / len(df)) * 100
    print(f"{diag:12s} : {count:3d} patients ({pct:.1f}%)")

# ===== TEMPERATURE MOYENNE PAR DIAGNOSTIC =====
print(f"\n--- Temperature moyenne par diagnostic ---")
temp_by_diag = df.groupby("diagnostic")["temperature"].mean()
for diag, temp in temp_by_diag.items():
    print(f"{diag:12s} : {temp:.1f} C")

# ===== PIED DE PAGE =====
print(f"\n{'=' * 50}")
print("Exploration terminee !")
print("Prochain lab : entrainer un modele ML")
print(f"{'=' * 50}")