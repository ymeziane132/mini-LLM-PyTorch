
from pathlib import Path
import torch
import torch.nn as nn

from fonction import (
    charger_texte,
    creer_dictionnaires,
    encoder_texte,
    decoder_tokens,
    preparer_donnees,
    entrainer_modele,
    evaluer_modele,
    generer_texte
)


################# Chargement du texte #################

chemin = Path("C:\\Users\\User\\Desktop\\projet_llm\\data\\text.txt")

texte = charger_texte(chemin)


################# Tokenisation #################

# Création des dictionnaires caractère -> identifiant et inversement
dictionnaire, dictionnaire_inverse = creer_dictionnaires(texte)


# Encodage du texte en identifiants numériques
token = encoder_texte(texte, dictionnaire)


# Vérification du décodage
text_decode = decoder_tokens(token, dictionnaire_inverse)

print("Décodage correct :", texte == text_decode)


################# Préparation des données #################

# Conversion en tenseur et séparation entraînement / validation
train_data, val_data = preparer_donnees(token)

print("Taille entraînement :", train_data.shape)
print("Taille validation :", val_data.shape)


################# Paramètres du modèle #################

block_size = 4
batch_size = 8
nb_iterations = 1000
learning_rate = 0.01

vocab_size = len(dictionnaire)


################# Création du modèle #################

# Création de la table de prédiction
table = nn.Embedding(vocab_size, vocab_size)

# Création de l'optimiseur
optimizer = torch.optim.AdamW(
    table.parameters(),
    lr=learning_rate
)


################# Entraînement du modèle #################

table = entrainer_modele(
    table,
    optimizer,
    train_data,
    batch_size,
    block_size,
    vocab_size,
    nb_iterations
)


################# Évaluation du modèle #################

resultats = evaluer_modele(
    table,
    train_data,
    val_data,
    batch_size,
    block_size,
    vocab_size
)

print("Loss entraînement :", resultats["train"])
print("Loss validation :", resultats["val"])

################# Génération de texte #################

# Choix du caractère de départ
caractere_depart = "L"

# Récupération de son identifiant
token_depart = dictionnaire[caractere_depart]

# Génération de 100 nouveaux caractères
tokens_generes = generer_texte(
    table,
    token_depart,
    100
)

# Décodage des tokens en texte
texte_genere = decoder_tokens(
    tokens_generes,
    dictionnaire_inverse
)

print("Texte généré :")
print(texte_genere)
