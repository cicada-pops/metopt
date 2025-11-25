"""
Simplex Method Implementation for Linear Programming Problems
Supports two-phase simplex method for solving general LP problems
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from enum import Enum


class ProblemType(Enum):
    """Type of optimization problem"""
    MAXIMIZE = "max"
    MINIMIZE = "min"


class ConstraintType(Enum):
    """Type of constraint"""
    LEQ = "<="  # Less than or equal
    EQ = "="    # Equal
    GEQ = ">="  # Greater than or equal


class SolutionStatus(Enum):
    """Status of the solution"""
    OPTIMAL = "optimal"
    UNBOUNDED = "unbounded"
    INFEASIBLE = "infeasible"
    NO_SOLUTION = "no_solution"


class SimplexSolver:
    """
    Two-phase Simplex Method solver for Linear Programming problems
    """
    
    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize the simplex solver
        
        Args:
            tolerance: Numerical tolerance for comparisons
        """
        self.tolerance = tolerance
        self.phase = 0
        self.iteration = 0
        
    def solve(self, c: np.ndarray, A: np.ndarray, b: np.ndarray, 
              constraint_types: List[ConstraintType], 
              problem_type: ProblemType = ProblemType.MAXIMIZE,
              verbose: bool = True) -> Dict:
        """
        Solve a linear programming problem using two-phase simplex method
        
        Args:
            c: Coefficients of objective function
            A: Constraint matrix
            b: Right-hand side values
            constraint_types: Types of constraints
            problem_type: MAXIMIZE or MINIMIZE
            verbose: Print detailed information
            
        Returns:
            Dictionary with solution information
        """
        if verbose:
            print("="*70)
            print("РЕШЕНИЕ ЗАДАЧИ ЛИНЕЙНОГО ПРОГРАММИРОВАНИЯ")
            print("="*70)
            print(f"\nТип задачи: {problem_type.value.upper()}")
            print(f"Количество переменных: {len(c)}")
            print(f"Количество ограничений: {len(b)}")
        
        # Store if we're minimizing (to convert back at the end)
        is_minimizing = (problem_type == ProblemType.MINIMIZE)
        
        # Convert to canonical form
        canonical = self._to_canonical_form(c, A, b, constraint_types, problem_type, verbose)
        
        if canonical is None:
            return {
                'status': SolutionStatus.INFEASIBLE,
                'message': 'Некорректная постановка задачи'
            }
        
        A_canon, b_canon, c_canon, n_original = canonical
        
        # Phase 1: Solve auxiliary problem to find initial feasible solution
        if verbose:
            print("\n" + "="*70)
            print("ФАЗА 1: РЕШЕНИЕ ВСПОМОГАТЕЛЬНОЙ ЗАДАЧИ")
            print("="*70)
        
        self.phase = 1
        phase1_result = self._phase1(A_canon, b_canon, verbose)
        
        if phase1_result['status'] != SolutionStatus.OPTIMAL:
            return phase1_result
        
        # Check if auxiliary problem has optimal value = 0
        if abs(phase1_result['objective']) > self.tolerance:
            if verbose:
                print(f"\nВспомогательная задача имеет оптимальное значение {-phase1_result['objective']:.6f} ≠ 0")
                print("ЗАДАЧА НЕСОВМЕСТНА (нет допустимых решений)")
            return {
                'status': SolutionStatus.INFEASIBLE,
                'message': 'Задача не имеет допустимых решений',
                'auxiliary_value': -phase1_result['objective']
            }
        
        # Phase 2: Solve original problem
        if verbose:
            print("\n" + "="*70)
            print("ФАЗА 2: РЕШЕНИЕ ОСНОВНОЙ ЗАДАЧИ")
            print("="*70)
        
        self.phase = 2
        phase2_result = self._phase2(
            A_canon,
            b_canon,
            c_canon,
            phase1_result['solution'][:A_canon.shape[1]],
            phase1_result['basis'],
            n_original,
            verbose
        )
        
        if phase2_result['status'] == SolutionStatus.OPTIMAL:
            # Extract original variables
            x_opt = phase2_result['solution'][:n_original]
            phase2_result['solution'] = x_opt
            
            # If we were minimizing, flip the objective back
            if is_minimizing:
                phase2_result['objective'] = -phase2_result['objective']
            
            if verbose:
                print("\n" + "="*70)
                print("ИТОГОВОЕ РЕШЕНИЕ")
                print("="*70)
                print(f"Статус: ОПТИМАЛЬНОЕ РЕШЕНИЕ НАЙДЕНО")
                print(f"\nОптимальная точка:")
                for i, val in enumerate(x_opt, 1):
                    print(f"  x{i} = {val:.6f}")
                print(f"\nЗначение целевой функции: Z = {phase2_result['objective']:.6f}")
        
        return phase2_result
    
    def _to_canonical_form(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
                          constraint_types: List[ConstraintType],
                          problem_type: ProblemType,
                          verbose: bool) -> Optional[Tuple]:
        """
        Convert LP problem to canonical form:
        - All constraints as equalities
        - All variables >= 0
        - Maximization problem (convert if minimization)
        
        Returns:
            Tuple of (A_canonical, b_canonical, c_canonical, n_original_vars)
        """
        if verbose:
            print("\n" + "-"*70)
            print("ПРИВЕДЕНИЕ К КАНОНИЧЕСКОМУ ВИДУ")
            print("-"*70)
        
        m, n = A.shape
        A = A.astype(float).copy()
        b = b.astype(float).copy()
        c = c.astype(float).copy()
        
        # Check for negative b values
        for i in range(m):
            if b[i] < -self.tolerance:
                A[i] = -A[i]
                b[i] = -b[i]
                # Flip constraint type
                if constraint_types[i] == ConstraintType.LEQ:
                    constraint_types[i] = ConstraintType.GEQ
                elif constraint_types[i] == ConstraintType.GEQ:
                    constraint_types[i] = ConstraintType.LEQ
        
        # Convert to maximization if minimizing (multiply c by -1)
        if problem_type == ProblemType.MINIMIZE:
            c = -c
            if verbose:
                print("Преобразование минимизации в максимизацию (умножение целевой функции на -1)")
        
        # Add slack/surplus variables
        A_canon = A.copy()
        c_canon = c.copy()
        
        slack_count = 0
        surplus_count = 0
        
        for i, ctype in enumerate(constraint_types):
            if ctype == ConstraintType.LEQ:
                # Add slack variable
                col = np.zeros((m, 1))
                col[i, 0] = 1
                A_canon = np.hstack([A_canon, col])
                c_canon = np.append(c_canon, 0)
                slack_count += 1
                if verbose:
                    print(f"Ограничение {i+1}: добавлена дополнительная переменная (slack)")
            
            elif ctype == ConstraintType.GEQ:
                # Add surplus variable
                col = np.zeros((m, 1))
                col[i, 0] = -1
                A_canon = np.hstack([A_canon, col])
                c_canon = np.append(c_canon, 0)
                surplus_count += 1
                if verbose:
                    print(f"Ограничение {i+1}: добавлена избыточная переменная (surplus)")
        
        if verbose:
            print(f"\nВсего переменных после приведения к каноническому виду: {A_canon.shape[1]}")
            print(f"  - Исходных: {n}")
            print(f"  - Дополнительных (slack): {slack_count}")
            print(f"  - Избыточных (surplus): {surplus_count}")
        
        return A_canon, b, c_canon, n
    
    def _phase1(self, A: np.ndarray, b: np.ndarray, verbose: bool) -> Dict:
        """
        Phase 1: Solve auxiliary problem to find initial basic feasible solution
        
        Minimize sum of artificial variables (= maximize -sum)
        """
        m, n = A.shape
        
        # Add artificial variables for each constraint
        A_aux = np.hstack([A, np.eye(m)])
        
        # Objective: minimize sum of artificial variables
        # In canonical form (maximization): maximize negative sum
        c_aux = np.zeros(n + m)
        c_aux[n:] = -1  # Coefficients for artificial variables
        
        # Initial basis: artificial variables
        basis = list(range(n, n + m))
        
        if verbose:
            print(f"\nНачальный базис: искусственные переменные {[f'a{i+1}' for i in range(m)]}")
            print(f"Целевая функция фазы 1: минимизировать сумму искусственных переменных")
        
        # Solve using simplex method
        result = self._simplex(A_aux, b, c_aux, basis, verbose)
        
        return result
    
    def _phase2(self, A: np.ndarray, b: np.ndarray, c: np.ndarray, 
                initial_solution: np.ndarray, initial_basis: List[int],
                n_original: int, verbose: bool) -> Dict:
        """
        Phase 2: Solve original problem starting from feasible solution
        """
        m, n = A.shape
        
        # Filter basis to remove artificial variables
        basis = [idx for idx in initial_basis if idx < n]
        
        # If we lost some basis variables, we need to find replacements
        # This shouldn't happen if phase 1 worked correctly
        if len(basis) < m:
            if verbose:
                print("ПРЕДУПРЕЖДЕНИЕ: Некоторые искусственные переменные остались в базисе")
            # Try to find replacement basis variables
            for i in range(n):
                if i not in basis and len(basis) < m:
                    basis.append(i)
        
        if verbose:
            print(f"\nБазис для основной задачи: {[f'x{i+1}' for i in basis]}")
            print(f"Целевая функция: исходная задача")
        
        # Solve using simplex method with original objective
        result = self._simplex(A, b, c, basis, verbose)
        
        return result
    
    def _simplex(self, A: np.ndarray, b: np.ndarray, c: np.ndarray,
                 basis: List[int], verbose: bool) -> Dict:
        """
        Execute simplex algorithm
        
        Args:
            A: Constraint matrix
            b: Right-hand side
            c: Objective coefficients
            basis: Initial basis
            verbose: Print information
        """
        m, n = A.shape
        self.iteration = 0
        
        # Create initial tableau
        # Tableau has the form:
        # [A | I | b]
        # where I is the basis (we'll maintain it implicitly)
        
        A = A.astype(float).copy()
        b = b.astype(float).copy()
        c = c.astype(float).copy()
        
        # Make sure basis is valid
        basis = list(basis)
        if len(basis) != m:
            return {
                'status': SolutionStatus.NO_SOLUTION,
                'message': f'Некорректный базис: размер {len(basis)} != {m}'
            }
        
        while True:
            self.iteration += 1
            
            if self.iteration > 200:  # Prevent infinite loops
                return {
                    'status': SolutionStatus.NO_SOLUTION,
                    'message': 'Превышено максимальное количество итераций'
                }
            
            # Compute current solution
            solution = np.zeros(n)
            
            # Get basis matrix and compute basic solution
            B = A[:, basis]
            try:
                B_inv = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                return {
                    'status': SolutionStatus.NO_SOLUTION,
                    'message': 'Вырожденная базисная матрица'
                }
            
            x_B = B_inv @ b
            
            # Check feasibility
            if np.any(x_B < -self.tolerance):
                if verbose:
                    print(f"\nИтерация {self.iteration}: Решение недопустимо")
                # This shouldn't happen in a correct implementation
                return {
                    'status': SolutionStatus.NO_SOLUTION,
                    'message': 'Получено недопустимое базисное решение'
                }
            
            for i, idx in enumerate(basis):
                solution[idx] = max(0, x_B[i])  # Ensure non-negative
            
            # Compute reduced costs
            c_B = c[basis]
            pi = c_B @ B_inv  # Dual variables
            reduced_costs = c - pi @ A
            
            if verbose:
                print(f"\n--- Итерация {self.iteration} (Фаза {self.phase}) ---")
                print(f"Базис: {[f'x{i+1}' for i in basis]}")
                print(f"Базисное решение: {x_B}")
                obj_value = c_B @ x_B
                print(f"Значение целевой функции: {obj_value:.6f}")
            
            # Check optimality
            if np.all(reduced_costs <= self.tolerance):
                # Optimal solution found
                obj_value = c @ solution
                
                if verbose:
                    print(f"\nОПТИМАЛЬНОЕ РЕШЕНИЕ найдено на итерации {self.iteration}")
                    print(f"Значение целевой функции: {obj_value:.6f}")
                
                return {
                    'status': SolutionStatus.OPTIMAL,
                    'solution': solution,
                    'objective': obj_value,
                    'basis': basis,
                    'iterations': self.iteration
                }
            
            # Select entering variable (most positive reduced cost)
            entering = np.argmax(reduced_costs)
            
            if reduced_costs[entering] <= self.tolerance:
                # All reduced costs non-positive, optimal
                obj_value = c @ solution
                return {
                    'status': SolutionStatus.OPTIMAL,
                    'solution': solution,
                    'objective': obj_value,
                    'basis': basis,
                    'iterations': self.iteration
                }
            
            if verbose:
                print(f"Входящая переменная: x{entering+1} (редуцир. стоимость: {reduced_costs[entering]:.6f})")
            
            # Compute direction
            d_B = B_inv @ A[:, entering]
            
            # Minimum ratio test
            ratios = []
            for i in range(m):
                if d_B[i] > self.tolerance:
                    ratio = x_B[i] / d_B[i]
                    ratios.append((ratio, i))
            
            if len(ratios) == 0:
                # No positive direction element => unbounded
                if verbose:
                    print("\nЗАДАЧА НЕ ОГРАНИЧЕНА (unbounded)")
                    print("Нет положительных элементов в направлении")
                return {
                    'status': SolutionStatus.UNBOUNDED,
                    'message': 'Целевая функция не ограничена'
                }
            
            # Select leaving variable (minimum ratio)
            min_ratio, leaving_idx = min(ratios)
            leaving = basis[leaving_idx]
            
            if verbose:
                print(f"Выходящая переменная: x{leaving+1} (индекс: {leaving_idx}, отношение: {min_ratio:.6f})")
            
            # Update basis
            basis[leaving_idx] = entering
            
            if verbose:
                print(f"Новый базис: {[f'x{i+1}' for i in basis]}")


def parse_lp_problem(filename: str) -> Dict:
    """
    Parse LP problem from text file
    
    Expected format:
    MAXIMIZE or MINIMIZE
    c1 c2 c3 ...  (objective coefficients)
    m n  (number of constraints, number of variables)
    a11 a12 ... a1n type1 b1
    a21 a22 ... a2n type2 b2
    ...
    where type is <=, =, or >=
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    idx = 0
    
    # Problem type
    problem_type_str = lines[idx].upper()
    problem_type = ProblemType.MAXIMIZE if 'MAX' in problem_type_str else ProblemType.MINIMIZE
    idx += 1
    
    # Objective coefficients
    c = np.array([float(x) for x in lines[idx].split()])
    idx += 1
    
    # Dimensions
    m, n = map(int, lines[idx].split())
    idx += 1
    
    # Constraints
    A = []
    b = []
    constraint_types = []
    
    for i in range(m):
        parts = lines[idx].split()
        a_row = [float(x) for x in parts[:n]]
        A.append(a_row)
        
        ctype_str = parts[n]
        if ctype_str == '<=':
            ctype = ConstraintType.LEQ
        elif ctype_str == '=':
            ctype = ConstraintType.EQ
        elif ctype_str == '>=':
            ctype = ConstraintType.GEQ
        else:
            raise ValueError(f"Unknown constraint type: {ctype_str}")
        
        constraint_types.append(ctype)
        b.append(float(parts[n+1]))
        idx += 1
    
    return {
        'problem_type': problem_type,
        'c': np.array(c),
        'A': np.array(A),
        'b': np.array(b),
        'constraint_types': constraint_types
    }
