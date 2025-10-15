# Computerv1

## Description

**Computerv1** est un projet Python qui implémente un programme de calculateur, capable de résoudre des expressions mathématiques et d'afficher les résultats. Il s'agit d'un projet éducatif visant à illustrer la manipulation d'expressions, l'analyse syntaxique, et la résolution d'équations en Python.

## Fonctionnalités

- Lecture d'expressions mathématiques depuis l'entrée standard ou un fichier
- Analyse et validation syntaxique des expressions
- Calcul et affichage du résultat
- Gestion des erreurs (syntaxe, division par zéro, etc.)

## Structure du projet

- `computor.py` : Fichier principal contenant la logique du calculateur
- `test.sh` : Script shell pour lancer des tests automatiques
- `__pycache__/` : Dossier généré automatiquement par Python pour le cache des modules compilés

## Prérequis

- Python 3.8 ou supérieur

## Installation

Aucune installation particulière n'est requise. Clonez simplement le dépôt :

```bash
git clone <url-du-repo> <nom_du_repo>
cd <nom_du_repo>
```

## Utilisation

### Lancer le programme principal

```bash
python3 computor.py
```

Vous pouvez alors entrer une equation mathématique à résoudre, par exemple :

```
> X + 1 = 0
-1
```

### Lancer les tests

Un script de test est fourni pour valider le fonctionnement du programme :

```bash
bash test.sh
```


## Gestion des erreurs

Le programme affiche un message d'erreur clair en cas de :
- Syntaxe incorrecte
- Division par zéro
- Expression non supportée

## Auteurs

- Aurelien Fontaine

