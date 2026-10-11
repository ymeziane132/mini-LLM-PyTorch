# Mini Language Model — PyTorch

## Présentation du projet

Ce projet consiste à développer **un modèle de langage neuronal à partir de zéro (from scratch)** avec Python et PyTorch.

L'objectif est de comprendre les mécanismes fondamentaux du fonctionnement des modèles de langage : tokenisation, embeddings, prédiction de tokens, calcul de la fonction de perte et entraînement d'un réseau de neurones.

Le modèle actuel fonctionne au **niveau des caractères** : il apprend à prédire le caractère suivant à partir des caractères précédents et génère du texte de manière autorégressive.

Il s'agit d'un modèle pédagogique simplifié, et non d'un LLM de type GPT ou d'une architecture Transformer complète.

## Technologies utilisées

- **Python** : développement et traitement des données.
- **PyTorch** : construction et entraînement du modèle neuronal.
- **Git / GitHub** : versionnement du code et gestion du projet.

## Étapes réalisées

### 1. Préparation des données

- Chargement d'un corpus textuel depuis `data/text.txt`.
- Identification des caractères uniques pour construire le vocabulaire.
- Création des dictionnaires de correspondance caractère → identifiant et identifiant → caractère.
- Implémentation des fonctions d'encodage et de décodage.
- Conversion du texte en tenseurs PyTorch.
- Séparation des données en ensembles d'entraînement (80 %) et de validation (20 %).

### 2. Création des séquences d'entraînement

- Développement d'une fonction `get_batch()` permettant de générer des mini-lots de données.
- Utilisation de fenêtres de contexte de taille `block_size = 4`.
- Constitution de séquences d'entrée et de leurs cibles décalées d'un caractère.
- Utilisation d'un `batch_size = 8`.

### 3. Construction du modèle neuronal

- Implémentation d'une table d'embedding `nn.Embedding(vocab_size, vocab_size)`.
- Transformation des identifiants de caractères en scores de prédiction (*logits*).
- Calcul de la fonction de perte avec `cross_entropy`.
- Adaptation des dimensions des tenseurs pour calculer la perte sur l'ensemble des positions du batch.

### 4. Entraînement du modèle

- Implémentation d'une boucle d'entraînement avec PyTorch.
- Calcul de la perte et rétropropagation des gradients.
- Optimisation des paramètres du modèle.
- Suivi de l'évolution de la perte pendant l'entraînement.

### 5. Évaluation des performances

- Mesure de la perte sur les données d'entraînement et de validation.
- Comparaison des performances afin d'observer la capacité de généralisation du modèle.
- Identification d'un possible surapprentissage lorsque la perte de validation dépasse significativement celle d'entraînement.

### 6. Génération de texte

- Implémentation de la génération autorégressive.
- Prédiction et échantillonnage successifs de nouveaux caractères.
- Décodage des identifiants générés en texte lisible.

Exemple de génération obtenue :

```text
Le.
Le sor pochige chate matt sie.
Le cane dhan jomane can cr chapé.
Le chie chaoue crt mmet capomman
```

Le texte généré conserve certaines structures du corpus d'entraînement, mais manque encore de cohérence. Cette limite s'explique notamment par l'architecture très simple du modèle, qui ne prend pas encore en compte les dépendances contextuelles entre les caractères.

## Architecture actuelle

```text
Corpus textuel
      ↓
Tokenisation caractère par caractère
      ↓
Encodage en identifiants numériques
      ↓
Création des batchs (X, Y)
      ↓
Table d'embedding PyTorch
      ↓
Logits (scores des caractères)
      ↓
Cross-Entropy Loss
      ↓
Rétropropagation et optimisation
      ↓
Génération autorégressive
```

**Limitation actuelle :** la table d'embedding associe des scores fixes à chaque caractère. Le modèle ne combine pas encore les informations des caractères précédents pour prédire le suivant.

## Prochaines étapes

- [ ] **Intégrer le contexte** : faire dépendre les prédictions de plusieurs caractères précédents plutôt que du seul caractère courant.
- [ ] **Améliorer les embeddings** : séparer la dimension des représentations internes de la taille du vocabulaire.
- [ ] **Ajouter des embeddings positionnels** : permettre au modèle de représenter la position des tokens.
- [ ] **Implémenter le mécanisme de Self-Attention** : apprendre les relations entre les différents tokens d'une séquence.
- [ ] **Construire un bloc Transformer** : intégrer Multi-Head Attention, Feed-Forward et normalisation.
- [ ] **Améliorer l'entraînement** : tester différents hyperparamètres et comparer les pertes d'entraînement et de validation.
- [ ] **Entraîner sur un corpus plus volumineux** : améliorer la diversité et la qualité des textes générés.
- [ ] **Sauvegarder et recharger les paramètres** : utiliser `torch.save()` et `torch.load()`.

## Structure du projet

```text
projet_llm/
├── data/
│   └── text.txt
├── fonction.py
├── main.py
├── .gitignore
└── README.md
```

**`main.py`** : exécution des différentes étapes du modèle.

**`fonction.py`** : fonctions de préparation des données, d'entraînement et d'évaluation.

**`data/text.txt`** : corpus utilisé pour l'entraînement.

## Objectifs pédagogiques

Ce projet me permet de développer mes compétences en :

- Programmation Python et manipulation de tenseurs.
- Deep Learning et réseaux de neurones.
- Préparation de données pour l'apprentissage automatique.
- Fonctions de perte, optimisation et rétropropagation.
- Évaluation et compréhension du surapprentissage.
- Architecture et fonctionnement des modèles de langage.

## Utilisation de l'intelligence artificielle

Pour réaliser ce projet, je me suis appuyé sur **ChatGPT comme assistant pédagogique et technique**.

Son utilisation m'a notamment permis de :

- Comprendre les concepts de tokenisation, embeddings, tenseurs, logits et Cross-Entropy.
- Obtenir des explications sur les mécanismes d'entraînement des réseaux de neurones.
- Identifier et corriger certaines erreurs de programmation.
- Structurer progressivement les différentes étapes du projet.

L'objectif de cette démarche est de développer ma compréhension du Deep Learning à travers une implémentation pratique, en analysant le fonctionnement du code et les résultats du modèle.

## Statut du projet

**En cours de développement.**

Première version fonctionnelle : préparation des données, entraînement, évaluation et génération de texte.

Prochaine évolution majeure : introduction de la prise en compte du contexte, puis implémentation progressive des mécanismes d'attention.
