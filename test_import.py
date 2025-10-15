#!/usr/bin/env python3
"""
Test pour démontrer l'utilité de if __name__ == "__main__":
"""

print("=== Script test_import.py exécuté ===")
print(f"__name__ = '{__name__}'")

def main():
    print("Fonction main() appelée")

if __name__ == "__main__":
    print("Le fichier est exécuté directement")
    main()
else:
    print("Le fichier est importé comme module")

print("=== Fin du script test_import.py ===")
