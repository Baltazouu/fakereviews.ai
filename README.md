# Fakereviews.ai

## About the dataset : 

| Category             | Rating | Label | Text_ |
|----------------------|--------|-------|----------------------------------------------------------------------------------------------------------------|
| Home_and_Kitchen_5  | 5.0    | OR    | Love this! Well made, sturdy, and very comfortable. I love it! Very pretty.                                  |
| Home_and_Kitchen_5  | 5.0    | CG    | Love it, a great upgrade from the original. I've had mine for a couple of years.                             |

**Légend** :  
- **OR** = Original Reviews  
- **CG** = Computer Generated Fake Reviews  


## Setup 

**Create a venv**

``python3 -m venv fake_review_env``
``source mon_env/bin/activate``

**Install dependancies**

``pip install -r requirements.txt``

** Run script **
``python fakereview.py``

## 1. Visualisation des datasets

- Éxecuter le script **graphs** de cette manière `python scripts/graphs.py`

## 2. Fusion des datasets 

- Exécuter le script **merge_reviews** de cette manière  `python scripts/merge_reviews.py`

- Pour visualiser la nouvelle répartition des données éxécuter le script **graphs2** de la manière suivante `python scripts/merge_reviews.py`

- Pour consulter la répartition des avis en fonction de leur longueur éxecuter le script **graphs3** via cette commande `python scripts/graphs3.py`


## 3. Création & Entraînement du modèle

- Exécuter le script `python fakereview_model.py`
- Les modèles sont générés dans le répertoire **models**

## 4. Analyse de performance 

- Le script **fakereview_performance** utilise le dataset original (500 dernières lignes supprimées de l'entraînement) et teste les avis.
Pour visualiser les résultats exécuter cette commande `python scripts/fakereview_performance.py`



## 5. Test manuel
- Pour tester manuellement le modèle vous pouvez éxecuter le script **fakereview_test** : `python scripts/fakereview_test.py`
*Nous vous conseillons d'utiliser les 500 dernières lignes du dataset **fakereview_final.csv** pour vos tests.*


## 6. Apprentissage non supervisé

- Pour générer des modèles non supervisés executer le script python fakereview_model_unsupervised.py : `python scripts/fakereview_model_unsupervised.py `
- Pour générer le graphique de résultat : `python scripts/fakereview_unsupervised_performance.py`




