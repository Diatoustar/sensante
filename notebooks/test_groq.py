# notebooks/test_groq.py
# Test de l'API Groq avec Llama 3
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# 1. On force Python à remonter d'un dossier pour trouver le .env (depuis notebooks/ vers sensante/)
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

# 2. Récupération de la clé sans aucun espace dans les guillemets
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERREUR : GROQ_API_KEY non trouvee.")
    print(f"Le script a cherche le fichier .env ici : {env_path}")
    exit()

# Créer le client Groq
client = Groq(api_key=api_key)

# Premier appel : question simple (chaînes de caractères nettoyées des espaces superflus)
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": "Tu es un assistant medical senegalais. Reponds en francais simple. Maximum 3 phrases.",
        },
        {
            "role": "user",
            "content": "Quels sont les symptomes du paludisme?",
        },
    ],
    max_tokens=200,
    temperature=0.3,
)

# Afficher la reponse
print("=== Reponse de Llama 3 ===")
print(response.choices[0].message.content)
print(f"\nTokens utilises : {response.usage.total_tokens}")

# Test avec le format SenSante
response2 = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": """Tu es un assistant medical senegalais .

Tu recois un diagnostic et des donnees patient .
Explique le resultat en francais simple ,
comme un medecin parlerait a son patient .
Sois rassurant mais recommande une consultation .
Maximum 3 phrases .
Ne fais JAMAIS de diagnostic toi - meme .""",
        },
        {
            "role": "user",
            "content": """Patient : Femme , 28 ans , region Dakar
Symptomes : temperature 39.5 , toux , fatigue , maux de tete
Diagnostic du modele : paludisme ( probabilite 72%)
Explique ce resultat au patient . """,
        },
    ],
    max_tokens=200,
    temperature=0.3,
)

print("=== Explication SenSante === ")
print(response2.choices[0].message.content)