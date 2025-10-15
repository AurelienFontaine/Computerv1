#!/usr/bin/env python3
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 test_simple.py \"equation\"")
        sys.exit(1)
    print(f"Résolution de : {sys.argv[1]}")

# Version SANS if __name__ == "__main__":
main()
