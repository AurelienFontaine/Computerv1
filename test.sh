#!/bin/bash
# Test script for Computor v1

echo "===== Testing Computor v1 ====="
echo ""

echo "Test 1: Quadratic with positive discriminant"
python3 computor.py "5 * X^0 + 4 * X^1 - 9.3 * X^2 = 1 * X^0"
echo ""

echo "Test 2: Linear equation"
python3 computor.py "5 * X^0 + 4 * X^1 = 4 * X^0"
echo ""

echo "Test 3: Degree 3 (unsolvable)"
python3 computor.py "8 * X^0 - 6 * X^1 + 0 * X^2 - 5.6 * X^3 = 3 * X^0"
echo ""

echo "Test 4: Infinite solutions"
python3 computor.py "6 * X^0 = 6 * X^0"
echo ""

echo "Test 5: No solution"
python3 computor.py "10 * X^0 = 15 * X^0"
echo ""

echo "Test 6: Complex solutions"
python3 computor.py "1 * X^0 + 2 * X^1 + 5 * X^2 = 0"
echo ""

echo "===== All tests completed ====="

