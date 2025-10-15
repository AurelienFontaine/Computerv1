import sys
import re


class PolynomialSolver:
    def __init__(self, equation_str, verbose=False):
        self.equation_str = equation_str
        self.coefficients = {}
        self.verbose = verbose
        
    def validate_equation(self):
        """Validate equation syntax and vocabulary before parsing"""
        eq = self.equation_str.strip()
        
        # Check if empty
        if not eq:
            raise ValueError("Error: Empty equation")
        
        # Check for exactly one '=' sign
        if '=' not in eq:
            raise ValueError("Error: Missing '=' sign in equation")
        
        if eq.count('=') > 1:
            raise ValueError("Error: Multiple '=' signs found")
        
        # Check for valid characters only
        # Valid: digits, X/x, ^, +, -, *, =, ., spaces
        valid_chars = set('0123456789Xx^+-*=. \t')
        invalid_chars = set(eq) - valid_chars
        if invalid_chars:
            raise ValueError(f"Error: Invalid characters found: {', '.join(sorted(invalid_chars))}")
        
        # Check that both sides exist
        parts = eq.split('=')
        if not parts[0].strip():
            raise ValueError("Error: Left side of equation is empty")
        if not parts[1].strip():
            raise ValueError("Error: Right side of equation is empty")
        
        # Check for invalid patterns
        if '**' in eq or '++' in eq or '--' in eq:
            raise ValueError("Error: Invalid operator sequence")
        
        # Check for misplaced operators
        if eq.startswith('*') or eq.startswith('/') or eq.endswith('*') or eq.endswith('/'):
            raise ValueError("Error: Misplaced operator")
        
        return True
    
    def parse_equation(self):
        """Parse the equation and extract coefficients for each power of X"""
        # Validate first
        self.validate_equation()
        
        # Split equation by '='
        parts = self.equation_str.split('=')
        
        left_side, right_side = parts
        
        # Parse both sides
        try:
            left_coeffs = self._parse_side(left_side)
            right_coeffs = self._parse_side(right_side)
        except Exception as e:
            raise ValueError(f"Error parsing equation: {e}")
        
        all_powers = set(left_coeffs.keys()) | set(right_coeffs.keys())
        self.coefficients = {}
        
        for power in all_powers:
            left_val = left_coeffs.get(power, 0)
            right_val = right_coeffs.get(power, 0)
            result = left_val - right_val
            self.coefficients[power] = result
        
        # If no coefficients, it means 0 = 0
        if not self.coefficients:
            self.coefficients[0] = 0
    
    def _parse_side(self, side_str):
        """Parse one side of the equation and return dict of {power: coefficient}"""
        coeffs = {}
        
        # Normalize the string: make it uppercase for X
        side_str = side_str.replace('x', 'X')
        
        # Multiple patterns to support free-form input:
        # 1. Standard: "5 * X^2" or "5*X^2"
        # 2. Free-form: "5X^2" (no *)
        # 3. Implicit coefficient: "X^2" (coefficient = 1), "-X^2" (coefficient = -1)
        # 4. Implicit power: "5X" or "X" (power = 1)
        # 5. Constant: "5" (power = 0)
        
        patterns = [
            # Standard format: coefficient * X^power
            (r'([+-]?\s*\d+\.?\d*)\s*\*\s*X\s*\^\s*(\d+)', 'coeff_power'),
            # Free-form: coefficientX^power (no *)
            (r'([+-]?\s*\d+\.?\d*)\s*X\s*\^\s*(\d+)', 'coeff_power'),
            # Implicit coefficient: X^power or -X^power or +X^power
            (r'([+-])\s*X\s*\^\s*(\d+)', 'sign_power'),
            # Just X^power (positive, at start or after =)
            (r'(?:^|=)\s*X\s*\^\s*(\d+)', 'just_power'),
            # coefficient * X or coefficientX (power = 1)
            (r'([+-]?\s*\d+\.?\d*)\s*\*?\s*X(?!\^)', 'coeff_x'),
            # Just X or -X or +X (coefficient = ±1, power = 1)
            (r'([+-])\s*X(?!\^)', 'sign_x'),
            # Just X at start (coefficient = 1, power = 1)
            (r'(?:^|=)\s*X(?!\^)', 'just_x'),
            # Constant only (no X)
            (r'([+-]?\s*\d+\.?\d*)(?!\s*\*?\s*X)', 'constant'),
        ]
        
        # Track what we've already parsed to avoid duplicates
        parsed_positions = set()
        
        for pattern, pattern_type in patterns:
            for match in re.finditer(pattern, side_str):
                # Skip if we already parsed this position
                start, end = match.span()
                if any(start < p < end or p == start for p in parsed_positions):
                    continue
                
                # Mark this region as parsed
                for i in range(start, end):
                    parsed_positions.add(i)
                
                if pattern_type == 'coeff_power':
                    coeff_str = match.group(1).replace(' ', '')
                    power = int(match.group(2))
                    coefficient = float(coeff_str)
                elif pattern_type == 'sign_power':
                    sign = match.group(1)
                    power = int(match.group(2))
                    coefficient = 1.0 if sign == '+' else -1.0
                elif pattern_type == 'just_power':
                    power = int(match.group(1))
                    coefficient = 1.0
                elif pattern_type == 'coeff_x':
                    coeff_str = match.group(1).replace(' ', '')
                    coefficient = float(coeff_str)
                    power = 1
                elif pattern_type == 'sign_x':
                    sign = match.group(1)
                    coefficient = 1.0 if sign == '+' else -1.0
                    power = 1
                elif pattern_type == 'just_x':
                    coefficient = 1.0
                    power = 1
                elif pattern_type == 'constant':
                    coeff_str = match.group(1).replace(' ', '')
                    if coeff_str and coeff_str not in ['+', '-']:
                        try:
                            coefficient = float(coeff_str)
                            power = 0
                        except ValueError:
                            continue
                    else:
                        continue
                else:
                    continue
                
                if power in coeffs:
                    coeffs[power] += coefficient
                else:
                    coeffs[power] = coefficient
        
        return coeffs
    
    def get_reduced_form(self):
        if not self.coefficients:
            return "0 * X^0 = 0"
        
        # Find the actual degree (highest non-zero power)
        actual_degree = self.get_degree()
        terms = []
        
        # Show all terms from 0 to actual degree
        for power in range(actual_degree + 1):
            coeff = self.coefficients.get(power, 0)
            
            # Format coefficient (remove .0 for whole numbers)
            if coeff == int(coeff):
                coeff_str = str(int(coeff))
            else:
                coeff_str = str(coeff)
            
            if not terms:  # First term
                terms.append(f"{coeff_str} * X^{power}")
            else:  # Subsequent terms
                if coeff >= 0:
                    terms.append(f"+ {coeff_str} * X^{power}")
                else:
                    abs_coeff = abs(coeff)
                    if abs_coeff == int(abs_coeff):
                        abs_coeff_str = str(int(abs_coeff))
                    else:
                        abs_coeff_str = str(abs_coeff)
                    terms.append(f"- {abs_coeff_str} * X^{power}")
        
        return ' '.join(terms) + " = 0"
    
    def get_degree(self):
        """Return the degree of the polynomial"""
        if not self.coefficients:
            return 0
        
        # Find highest power with non-zero coefficient
        max_degree = 0
        for power, coeff in self.coefficients.items():
            if coeff != 0 and power > max_degree:
                max_degree = power
        
        return max_degree
    
    def solve(self):
        """Solve the equation based on its degree"""
        degree = self.get_degree()
        
        print(f"Reduced form: {self.get_reduced_form()}")
        print(f"Polynomial degree: {degree}")
        
        if degree > 2:
            print("The polynomial degree is strictly greater than 2, I can't solve.")
            return
        
        if degree == 0:
            self._solve_degree_0()
        elif degree == 1:
            self._solve_degree_1()
        elif degree == 2:
            self._solve_degree_2()
    
    def _solve_degree_0(self):
        """Solve constant equation (degree 0)"""
        const = self.coefficients.get(0, 0)
        
        if const == 0:
            print("Any real number is a solution.")
        else:
            print("No solution.")
    
    def _solve_degree_1(self):
        """Solve linear equation: a*X + b = 0"""
        a = self.coefficients.get(1, 0)  # Coefficient of X^1
        b = self.coefficients.get(0, 0)  # Coefficient of X^0
        
        if self.verbose:
            print(f"\nSolving linear equation: {a}*X + {b} = 0")
            print(f"Formula: X = -b/a")
            print(f"X = -({b})/{a}")
            print(f"X = {-b}/{a}")
        
        # Solution: X = -b/a
        solution = -b / a
        solution_str = self._format_solution(solution)
        print(f"The solution is: {solution_str}")
    
    def _solve_degree_2(self):
        """Solve quadratic equation: a*X^2 + b*X + c = 0"""
        a = self.coefficients.get(2, 0)  # Coefficient of X^2
        b = self.coefficients.get(1, 0)  # Coefficient of X^1
        c = self.coefficients.get(0, 0)  # Coefficient of X^0
        
        if self.verbose:
            print(f"\nSolving quadratic equation: {a}*X² + {b}*X + {c} = 0")
            print(f"Using quadratic formula: X = (-b ± √(b² - 4ac)) / 2a")
            print(f"\nCoefficients: a = {a}, b = {b}, c = {c}")
        
        # Calculate discriminant: b^2 - 4ac
        discriminant = b * b - 4 * a * c
        
        if self.verbose:
            print(f"Calculating discriminant: Δ = b² - 4ac")
            print(f"Δ = ({b})² - 4*({a})*({c})")
            print(f"Δ = {b*b} - {4*a*c}")
            print(f"Δ = {discriminant}")
        
        if discriminant > 0:
            print("Discriminant is strictly positive, the two solutions are:")

            sqrt_discriminant = self._sqrt(discriminant)
            
            if self.verbose:
                print(f"\n√Δ = √{discriminant} = {sqrt_discriminant}")
                print(f"\nX₁ = (-b + √Δ) / 2a = (-({b}) + {sqrt_discriminant}) / (2*{a})")
                print(f"X₁ = ({-b} + {sqrt_discriminant}) / {2*a}")
                print(f"X₁ = {-b + sqrt_discriminant} / {2*a}")
                
            x1 = (-b + sqrt_discriminant) / (2 * a)
            
            if self.verbose:
                print(f"X₁ = {x1}")
                print(f"\nX₂ = (-b - √Δ) / 2a = (-({b}) - {sqrt_discriminant}) / (2*{a})")
                print(f"X₂ = ({-b} - {sqrt_discriminant}) / {2*a}")
                print(f"X₂ = {-b - sqrt_discriminant} / {2*a}")
                
            x2 = (-b - sqrt_discriminant) / (2 * a)
            
            if self.verbose:
                print(f"X₂ = {x2}\n")
            
            # Try to format as exact fractions first, then fall back to decimal
            x1_str = self._format_quadratic_solution(a, b, c, discriminant, True)
            x2_str = self._format_quadratic_solution(a, b, c, discriminant, False)
            
            if x1_str is None:
                x1_str = self._format_solution(x1)
            if x2_str is None:
                x2_str = self._format_solution(x2)
            
            if x1 > x2:
                print(x1_str)
                print(x2_str)
            else:
                print(x2_str)
                print(x1_str)
        elif discriminant == 0:
            print("Discriminant is zero, the solution is:")

            if self.verbose:
                print(f"\nX = -b / 2a = -({b}) / (2*{a})")
                print(f"X = {-b} / {2*a}")
            
            x = -b / (2 * a)
            
            if self.verbose:
                print(f"X = {x}\n")
            
            x_str = self._format_solution(x)
            print(x_str)
        else:  
            print("Discriminant is strictly negative, the two complex solutions are:")
            
            if self.verbose:
                print(f"\n√(-Δ) = √{-discriminant} (imaginary)")
            
            real_part = -b / (2 * a)
            sqrt_discriminant = self._sqrt(-discriminant)
            imaginary_part = sqrt_discriminant / (2 * a)
            
            if self.verbose:
                print(f"Real part: -b/2a = -({b})/(2*{a}) = {real_part}")
                print(f"Imaginary part: √|Δ|/2a = √{-discriminant}/(2*{a}) = {sqrt_discriminant}/{2*a} = {imaginary_part}")
            
            # Format complex solutions with exact forms
            real_str = self._format_complex_real_part(real_part)
            imag_str = self._format_complex_imaginary_part(imaginary_part, -discriminant, 2 * a)
            
            if self.verbose:
                print(f"\nX₁ = {real_part} + {imaginary_part}i")
                print(f"X₂ = {real_part} - {imaginary_part}i\n")
            
            # Format the final complex number properly
            if real_str == "0":
                print(f"{imag_str}")
                print(f"-{imag_str}")
            else:
                print(f"{real_str} + {imag_str}")
                print(f"{real_str} - {imag_str}")
    
    def _sqrt(self, x):
        """Calculate square root using Newton's method (no math library!)"""
        if x == 0:
            return 0
        
        # Newton's method: x_n+1 = (x_n + S/x_n) / 2
        guess = x / 2.0
        epsilon = 1e-10
        
        while True:
            next_guess = (guess + x / guess) / 2.0
            if abs(next_guess - guess) < epsilon:
                return next_guess
            guess = next_guess
    
    def _gcd(self, a, b):
        """Calculate greatest common divisor using Euclidean algorithm"""
        a, b = abs(a), abs(b)
        while b:
            a, b = b, a % b
        return a
    
    
    def _format_solution(self, value):
        """Format a solution value as fraction if possible, otherwise as decimal"""
        epsilon = 1e-9
        
        # Check if it's close to an integer
        if abs(value - round(value)) < epsilon:
            return str(int(round(value)))
        
        # Try to express as a simple fraction
        # Try denominators from 2 to 100 for better precision
        for denom in range(2, 101):
            numer = value * denom
            if abs(numer - round(numer)) < epsilon:
                numer = int(round(numer))
                gcd = self._gcd(numer, denom)
                numer //= gcd
                denom //= gcd
                if denom == 1:
                    return str(numer)
                else:
                    
                    decimal_val = value
                    return f"{numer}/{denom} ({decimal_val})"
        
        # If no simple fraction found, return decimal only
        return str(value)
    
    def _format_complex_real_part(self, real_part):
        """Format the real part of complex solutions"""
        epsilon = 1e-9
        
        # Check if it's close to an integer
        if abs(real_part - round(real_part)) < epsilon:
            return str(int(round(real_part)))
        
        # Try to express as a simple fraction
        for denom in range(2, 101):
            numer = real_part * denom
            if abs(numer - round(numer)) < epsilon:
                numer = int(round(numer))
                gcd = self._gcd(numer, denom)
                numer //= gcd
                denom //= gcd
                if denom == 1:
                    return str(numer)
                else:
                    return f"{numer}/{denom}"
        
        # If no simple fraction found, return decimal
        return str(real_part)
    
    def _format_complex_imaginary_part(self, imaginary_part, discriminant_abs, denominator):
        """Format the imaginary part of complex solutions with exact square roots"""
        epsilon = 1e-9
        
        # Check if we can factor out a perfect square from the discriminant
        # For example: discriminant = 8, so √8 = 2√2
        sqrt_disc = self._sqrt(discriminant_abs)
        
        # Try to find if discriminant_abs is a multiple of a perfect square
        # Start from the largest possible factor to get the most simplified form
        for factor in range(int(sqrt_disc), 1, -1):
            if discriminant_abs % (factor * factor) == 0:
                remaining = int(discriminant_abs // (factor * factor))
                # Case: discriminant_abs = factor² * remaining, so √discriminant = factor√remaining
                coeff = factor / denominator
                if abs(coeff - round(coeff)) < epsilon:
                    coeff_int = int(round(coeff))
                    if remaining == 1:
                        # Perfect case: discriminant_abs = factor², so √discriminant = factor
                        if coeff_int == 1:
                            return "i"
                        elif coeff_int == -1:
                            return "-i"
                        else:
                            return f"{coeff_int}i"
                    else:
                        if coeff_int == 1:
                            return f"√{remaining}i"
                        elif coeff_int == -1:
                            return f"-√{remaining}i"
                        else:
                            return f"{coeff_int}√{remaining}i"
                else:
                    # Try to simplify the fraction
                    for d in range(2, 101):
                        n = coeff * d
                        if abs(n - round(n)) < epsilon:
                            n = int(round(n))
                            gcd = self._gcd(n, d)
                            n //= gcd
                            d //= gcd
                            if remaining == 1:
                                if d == 1:
                                    if n == 1:
                                        return "i"
                                    elif n == -1:
                                        return "-i"
                                    else:
                                        return f"{n}i"
                                else:
                                    return f"{n}i/{d}"
                            else:
                                if d == 1:
                                    if n == 1:
                                        return f"√{remaining}i"
                                    elif n == -1:
                                        return f"-√{remaining}i"
                                    else:
                                        return f"{n}√{remaining}i"
                                else:
                                    return f"{n}√{remaining}i/{d}"
        
        # If no perfect square factorization found, try simple fraction
        if abs(imaginary_part - round(imaginary_part)) < epsilon:
            imag_int = int(round(imaginary_part))
            if imag_int == 1:
                return "i"
            elif imag_int == -1:
                return "-i"
            else:
                return f"{imag_int}i"
        
        for denom in range(2, 101):
            numer = imaginary_part * denom
            if abs(numer - round(numer)) < epsilon:
                numer = int(round(numer))
                gcd = self._gcd(numer, denom)
                numer //= gcd
                denom //= gcd
                if denom == 1:
                    if numer == 1:
                        return "i"
                    elif numer == -1:
                        return "-i"
                    else:
                        return f"{numer}i"
                else:
                    return f"{numer}i/{denom}"
        
        # If no simple form found, return decimal
        return str(imaginary_part)
    
    def _format_quadratic_solution(self, a, b, c, discriminant, is_positive):
        """Format quadratic solution with exact form when possible"""
        if discriminant < 0:
            # Complex case - handled separately
            return None
            
        sqrt_disc = self._sqrt(discriminant)
        
        # Check if discriminant is a perfect square
        sqrt_disc_int = round(sqrt_disc)
        if abs(sqrt_disc - sqrt_disc_int) < 1e-9:
            # Perfect square case
            if is_positive:
                numer = -b + sqrt_disc_int
            else:
                numer = -b - sqrt_disc_int
            denom = 2 * a
            
            # Simplify the fraction
            gcd = self._gcd(numer, denom)
            numer //= gcd
            denom //= gcd
            
            if denom == 1:
                # Convert to int to remove .0
                return str(int(numer))
            elif denom == -1:
                # Convert to int to remove .0
                return str(int(-numer))
            else:
                # Convert to int to remove .0 if possible
                if abs(numer - round(numer)) < 1e-9 and abs(denom - round(denom)) < 1e-9:
                    return f"{int(round(numer))}/{int(round(denom))}"
                else:
                    return f"{numer}/{denom}"
        
        
        numer = -b
        denom = 2 * a
        
        # Find common factors
        gcd_val = self._gcd(abs(numer), abs(denom))
        if gcd_val > 1:
            numer //= gcd_val
            denom //= gcd_val
            sqrt_disc = sqrt_disc / gcd_val
        
        
        # General case: try to format as (integer ± √integer) / integer
        if abs(numer - round(numer)) < 1e-9 and abs(denom - round(denom)) < 1e-9:
            numer_int = int(round(numer))
            denom_int = int(round(denom))
            
            # Check if sqrt_disc is close to √integer
            sqrt_disc_squared = sqrt_disc * sqrt_disc
            if abs(sqrt_disc_squared - round(sqrt_disc_squared)) < 1e-9:
                sqrt_val = int(round(sqrt_disc_squared))
                
                # Special case: if numerator is 0, just show ±√val
                if numer_int == 0:
                    if is_positive:
                        return f"√{sqrt_val}"
                    else:
                        return f"-√{sqrt_val}"
                # Special case: if denominator is 1, show without fraction
                elif denom_int == 1:
                    if is_positive:
                        return f"{numer_int} + √{sqrt_val}"
                    else:
                        return f"{numer_int} - √{sqrt_val}"
                else:
                    if is_positive:
                        return f"({numer_int} + √{sqrt_val}) / {denom_int}"
                    else:
                        return f"({numer_int} - √{sqrt_val}) / {denom_int}"
        
        # If no special case found, return decimal approximation
        if is_positive:
            solution = (-b + sqrt_disc) / (2 * a)
        else:
            solution = (-b - sqrt_disc) / (2 * a)
        
        # Format the solution properly (remove .0 for integers)
        if abs(solution - round(solution)) < 1e-9:
            return str(int(round(solution)))
        else:
            return str(solution)


def main():
    # Check for verbose flag
    verbose = False
    args = sys.argv[1:]
    
    if '-v' in args or '--verbose' in args:
        verbose = True
        args = [arg for arg in args if arg not in ['-v', '--verbose']]
    
    if len(args) != 1:
        print("Usage: ./computor.py [-v|--verbose] \"equation\"")
        print("Example: ./computor.py \"5 * X^0 + 4 * X^1 - 9.3 * X^2 = 1 * X^0\"")
        print("         ./computor.py -v \"X^2 + 2X + 1 = 0\"")
        sys.exit(1)
    
    equation = args[0]
    
    try:
        solver = PolynomialSolver(equation, verbose=verbose)
        solver.parse_equation()
        solver.solve()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

