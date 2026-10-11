import torch
import torch.nn as nn


################# Chargement du texte #################

def charger_texte(chemin):
    # Ouverture et lecture du fichier texte
    with chemin.open("r", encoding="utf-8") as fichier:
        texte = fichier.read()

    return texte


################# Création des dictionnaires #################

def creer_dictionnaires(texte):
    # Récupération des caractères uniques dans l'ordre alphabétique
    caracteres = sorted(set(texte))

    dictionnaire = {}
    dictionnaire_inverse = {}

    # Association des caractères à leurs identifiants
    for i, caractere in enumerate(caracteres):
        dictionnaire[caractere] = i
        dictionnaire_inverse[i] = caractere

    return dictionnaire, dictionnaire_inverse


################# Encodage du texte #################

def encoder_texte(texte, dictionnaire):
    # Transformation de chaque caractère en identifiant numérique
    token = []

    for caractere in texte:
        token.append(dictionnaire[caractere])

    return token


################# Décodage des tokens #################

def decoder_tokens(token, dictionnaire_inverse):
    # Reconstruction du texte à partir des identifiants numériques
    text_decode = ""

    for nombre in token:
        text_decode += dictionnaire_inverse[nombre]

    return text_decode


################# Préparation des données #################

def preparer_donnees(token):
    # Conversion des tokens en tenseur PyTorch
    data = torch.tensor(token, dtype=torch.long)

    # Calcul de la séparation 80 % / 20 %
    n = int(0.8 * len(data))

    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data


# Génère un batch aléatoire de séquences d'entrée et de cibles
def get_batch(data, batch_size, block_size):

    # Génération des positions de départ aléatoires
    indices = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,)
    )

    x_batch = []
    y_batch = []

    # Extraction des séquences
    for i in indices:
        x_sequence = data[i:i + block_size]
        y_sequence = data[i + 1:i + block_size + 1]

        x_batch.append(x_sequence)
        y_batch.append(y_sequence)

    # Regroupement des séquences en tenseurs
    x = torch.stack(x_batch)
    y = torch.stack(y_batch)

    return x, y


################# Entraînement du modèle #################

def entrainer_modele(table, optimizer, train_data,
                    batch_size, block_size, vocab_size,
                    nb_iterations):

    for step in range(nb_iterations):

        # Récupération d'un nouveau batch
        x, y = get_batch(train_data, batch_size, block_size)

        # Calcul des prédictions
        logits = table(x)

        # Adaptation des dimensions
        logits = logits.reshape(-1, vocab_size)
        y_flat = y.reshape(-1)

        # Calcul de l'erreur moyenne
        loss = nn.functional.cross_entropy(logits, y_flat)

        # Réinitialisation des gradients
        optimizer.zero_grad()

        # Calcul des gradients
        loss.backward()

        # Mise à jour des paramètres
        optimizer.step()

        # Affichage de la loss toutes les 100 itérations
        if step % 100 == 0:
            print(f"Étape {step} : Loss = {loss.item():.4f}")

    return table

################# Évaluation du modèle #################

def evaluer_modele(table, train_data, val_data,
                   batch_size, block_size, vocab_size):

    resultats = {}

    # Désactivation du calcul des gradients
    with torch.no_grad():

        # Évaluation sur les données d'entraînement et de validation
        for nom, data in [("train", train_data), ("val", val_data)]:

            losses = []

            # Calcul de la loss sur 20 batches
            for i in range(20):

                x, y = get_batch(data, batch_size, block_size)

                logits = table(x)
                logits = logits.reshape(-1, vocab_size)
                y_flat = y.reshape(-1)

                loss = nn.functional.cross_entropy(logits, y_flat)

                # Enregistrement de la loss du batch
                losses.append(loss.item())

            # Moyenne des 20 losses
            resultats[nom] = sum(losses) / len(losses)

    return resultats

################# Génération de texte #################

def generer_texte(table, token_depart, nb_caracteres):

    # Création du tenseur contenant le premier token
    tokens = torch.tensor([token_depart], dtype=torch.long)

    # Désactivation du calcul des gradients
    with torch.no_grad():

        for i in range(nb_caracteres):

            # Prédiction à partir du dernier token
            logits = table(tokens[-1:])

            # Conversion des scores en probabilités
            probabilites = torch.softmax(logits, dim=-1)

            # Sélection aléatoire selon les probabilités
            prochain_token = torch.multinomial(
                probabilites[0],
                num_samples=1
            )

            # Ajout du nouveau token à la séquence
            tokens = torch.cat((tokens, prochain_token))

    return tokens.tolist()