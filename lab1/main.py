"""
Основная программа для решения задачи линейного программирования
"""

import numpy as np
from simplex_solver import SimplexSolver, parse_lp_problem, ProblemType, ConstraintType


def find_all_integer_optima(c, A, b, constraint_types, optimal_value, tolerance=1e-6):
    """
    Найти все целочисленные оптимальные точки
    
    Args:
        c: коэффициенты целевой функции
        A: матрица ограничений
        b: правые части
        constraint_types: типы ограничений
        optimal_value: оптимальное значение целевой функции
        tolerance: точность
    
    Returns:
        Список целочисленных оптимальных точек
    """
    n = len(c)
    integer_solutions = []
    
    # Определить границы поиска (грубая оценка)
    max_val = int(max(b) + 1)
    
    # Перебор всех целочисленных точек в разумных границах
    from itertools import product
    
    # Ограничить поиск разумными пределами
    ranges = [range(0, min(max_val, 20)) for _ in range(n)]
    
    for point in product(*ranges):
        x = np.array(point, dtype=float)
        
        # Проверить допустимость
        feasible = True
        for i, ctype in enumerate(constraint_types):
            val = A[i] @ x
            if ctype == ConstraintType.LEQ:
                if val > b[i] + tolerance:
                    feasible = False
                    break
            elif ctype == ConstraintType.EQ:
                if abs(val - b[i]) > tolerance:
                    feasible = False
                    break
            elif ctype == ConstraintType.GEQ:
                if val < b[i] - tolerance:
                    feasible = False
                    break
        
        if not feasible:
            continue
        
        # Проверить оптимальность
        z = c @ x
        if abs(z - optimal_value) < tolerance:
            integer_solutions.append(x)
    
    return integer_solutions


def solve_variant_problem():
    """
    Решить задачу по варианту:
    Максимизировать Z = x1 + 3*x2 + 2*x3 + x4
    При условиях:
        x1 + x2 + 2*x4 <= 8
        x2 + x3 + x4 = 6
        2*x1 + x3 >= 2
        x1, x2, x3, x4 >= 0
    """
    print("="*70)
    print("РЕШЕНИЕ ЗАДАЧИ ЛИНЕЙНОГО ПРОГРАММИРОВАНИЯ")
    print("="*70)
    
    print("\nИСХОДНАЯ ЗАДАЧА:")
    print("-"*70)
    print("Максимизировать: Z = x1 + 3*x2 + 2*x3 + x4")
    print("\nПри условиях:")
    print("  1) x1 + x2 + 2*x4 <= 8")
    print("  2) x2 + x3 + x4 = 6")
    print("  3) 2*x1 + x3 >= 2")
    print("  4) x1, x2, x3, x4 >= 0")
    
    # Данные задачи
    c = np.array([1, 3, 2, 1])  # Коэффициенты целевой функции
    A = np.array([
        [1, 1, 0, 2],   # x1 + x2 + 2*x4 <= 8
        [0, 1, 1, 1],   # x2 + x3 + x4 = 6
        [2, 0, 1, 0]    # 2*x1 + x3 >= 2
    ])
    b = np.array([8, 6, 2])
    constraint_types = [
        ConstraintType.LEQ,
        ConstraintType.EQ,
        ConstraintType.GEQ
    ]
    
    # Решение
    print("\n" + "="*70)
    print("РЕШЕНИЕ СИМПЛЕКС-МЕТОДОМ")
    print("="*70)
    
    solver = SimplexSolver()
    result = solver.solve(c, A, b, constraint_types, ProblemType.MAXIMIZE, verbose=False)
    
    if result['status'].value == 'optimal':
        print(f"\nСтатус: ОПТИМАЛЬНОЕ РЕШЕНИЕ НАЙДЕНО")
        print(f"Количество итераций: {result['iterations']}")
        print(f"\nОдна из оптимальных точек:")
        for i, val in enumerate(result['solution'], 1):
            print(f"  x{i} = {val:.6f}")
        print(f"\nМаксимальное значение целевой функции: Z* = {result['objective']:.6f}")
        
        # Найти все целочисленные оптимальные решения
        print("\n" + "="*70)
        print("ВСЕ ЦЕЛОЧИСЛЕННЫЕ ОПТИМАЛЬНЫЕ ТОЧКИ")
        print("="*70)
        
        integer_solutions = find_all_integer_optima(c, A, b, constraint_types, result['objective'])
        
        if integer_solutions:
            print(f"\nНайдено целочисленных оптимальных точек: {len(integer_solutions)}")
            print()
            for i, sol in enumerate(integer_solutions, 1):
                x_int = sol.astype(int)
                z_val = c @ sol
                print(f"{i}. x = ({x_int[0]}, {x_int[1]}, {x_int[2]}, {x_int[3]}), Z = {z_val:.0f}")
        else:
            print("\nЦелочисленных оптимальных решений не найдено.")
        
    elif result['status'].value == 'unbounded':
        print(f"\nСтатус: ЗАДАЧА НЕ ОГРАНИЧЕНА")
        print(f"Причина: {result['message']}")
        
    elif result['status'].value == 'infeasible':
        print(f"\nСтатус: ЗАДАЧА НЕ ИМЕЕТ ДОПУСТИМЫХ РЕШЕНИЙ")
        print(f"Причина: {result['message']}")
        
    else:
        print(f"\nСтатус: {result['status'].value.upper()}")
        print(f"Сообщение: {result.get('message', 'Нет дополнительной информации')}")
    
    print("\n" + "="*70)
    
    return result


def solve_from_file(filename: str = "input.txt"):
    """
    Решить задачу из входного файла
    
    Формат файла:
        MAXIMIZE или MINIMIZE
        c1 c2 c3 ...
        m n
        a11 a12 ... a1n тип1 b1
        a21 a22 ... a2n тип2 b2
        ...
    где тип: <=, =, >=
    """
    print(f"Чтение задачи из файла: {filename}\n")
    
    problem = parse_lp_problem(filename)
    
    print("ИСХОДНАЯ ЗАДАЧА:")
    print("-"*70)
    print(f"Тип: {problem['problem_type'].value.upper()}")
    print(f"Переменных: {len(problem['c'])}")
    print(f"Ограничений: {len(problem['b'])}")
    
    # Решение
    solver = SimplexSolver()
    result = solver.solve(
        problem['c'],
        problem['A'],
        problem['b'],
        problem['constraint_types'],
        problem['problem_type'],
        verbose=False
    )
    
    if result['status'].value == 'optimal':
        print(f"\nСтатус: ОПТИМАЛЬНОЕ РЕШЕНИЕ НАЙДЕНО")
        print(f"\nОптимальная точка:")
        for i, val in enumerate(result['solution'], 1):
            print(f"  x{i} = {val:.6f}")
        
        z_val = result['objective']
        if problem['problem_type'] == ProblemType.MINIMIZE:
            z_val = -z_val
        print(f"\nЗначение целевой функции: Z = {z_val:.6f}")
    
    return result


if __name__ == "__main__":
    # Решение задачи по варианту
    result = solve_variant_problem()
    
    # Или решить задачу из файла:
    # result = solve_from_file("input.txt")
