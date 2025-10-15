#!/usr/bin/env python3
"""
Test pour démontrer ce qui se passe avec et sans if __name__ == "__main__":
"""

import sys

def main():
    print(f"Arguments reçus : {sys.argv}")
    if len(sys.argv) != 2:
        print("Erreur : Il faut exactement 1 argument")
        sys.exit(1)
    print(f"Traitement de : {sys.argv[1]}")

class MaClasse:
    def __init__(self, value):
        self.value = value
    def afficher(self):
        print(f"Valeur : {self.value}")

# Version SANS if __name__ == "__main__":
# Décommentez la ligne suivante pour tester :
main()
