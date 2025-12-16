# -*- coding: utf-8 -*-
"""
Оптимизация инвестиционного портфеля методом динамического программирования
с различными критериями принятия решений в условиях неопределённости.
Интерактивная версия с меню.
"""

from enum import Enum
from itertools import product
from typing import List, Tuple, Dict


class DecisionCriterion(Enum):
    """Критерии принятия решений в условиях неопределённости"""
    BAYES = "Байеса (математическое ожидание)"
    WALD = "Вальда (максимин)"
    SAVAGE = "Сэвиджа (минимакс сожалений)"
    HURWICZ = "Гурвица (оптимизм-пессимизм)"
    LAPLACE = "Лапласа (равновероятность)"


# Глобальные параметры задачи
INIT_STATE = [100.0, 800.0, 400.0, 600.0]  # ЦБ1, ЦБ2, Деп, Своб.средства
DELTA = [25, 200, 100]  # Шаги управления: ЦБ1, ЦБ2, Деп

# Коэффициенты изменения: (вероятность, r_цб1, r_цб2, r_деп)
STAGES_DATA = [
    # Этап 1
    [(0.60, 1.20, 1.10, 1.07), (0.30, 1.05, 1.02, 1.03), (0.10, 0.80, 0.95, 1.00)],
    # Этап 2
    [(0.30, 1.40, 1.15, 1.01), (0.20, 1.05, 1.00, 1.00), (0.50, 0.60, 0.90, 1.00)],
    # Этап 3
    [(0.40, 1.15, 1.12, 1.05), (0.40, 1.05, 1.01, 1.01), (0.20, 0.70, 0.94, 1.00)],
]

SCENARIO_NAMES = ["Благоприятная", "Нейтральная", "Негативная"]


def get_controls(state: List[float]) -> List[Tuple[int, int, int]]:
    """Генерация допустимых управлений (ограниченный набор для скорости)"""
    cb1, cb2, dep, cash = state
    controls = []
    
    # Базовый диапазон: от -1 до +1 пакета
    for u1 in range(-1, 2):
        for u2 in range(-1, 2):
            for u3 in range(-1, 2):
                new_cb1 = cb1 + u1 * DELTA[0]
                new_cb2 = cb2 + u2 * DELTA[1]
                new_dep = dep + u3 * DELTA[2]
                cost = u1 * DELTA[0] + u2 * DELTA[1] + u3 * DELTA[2]
                new_cash = cash - cost
                
                if new_cb1 >= 0 and new_cb2 >= 0 and new_dep >= 0 and new_cash >= 0:
                    controls.append((u1, u2, u3))
    
    # Агрессивные варианты
    max_u1 = int(cash // DELTA[0])
    if max_u1 > 1:
        controls.append((min(max_u1, 4), 0, 0))
    
    max_u2 = int(cash // DELTA[1])
    if max_u2 > 1:
        controls.append((0, min(max_u2, 2), 0))
    
    max_u3 = int(cash // DELTA[2])
    if max_u3 > 1:
        controls.append((0, 0, min(max_u3, 3)))
    
    return list(set(controls))


def apply_control(state: List[float], u: Tuple[int, int, int]) -> List[float]:
    """Применить управление к состоянию"""
    cost = u[0] * DELTA[0] + u[1] * DELTA[1] + u[2] * DELTA[2]
    return [state[0] + u[0] * DELTA[0], state[1] + u[1] * DELTA[1],
            state[2] + u[2] * DELTA[2], state[3] - cost]


def apply_scenario(state: List[float], scenario: Tuple) -> List[float]:
    """Применить сценарий к состоянию"""
    _, r1, r2, r3 = scenario
    return [state[0] * r1, state[1] * r2, state[2] * r3, state[3]]


def total_value(state: List[float]) -> float:
    """Общая стоимость портфеля"""
    return sum(state)


def evaluate_by_criterion(values: List[float], probs: List[float], 
                          criterion: DecisionCriterion, alpha: float = 0.5) -> float:
    """Оценка по критерию принятия решений"""
    if criterion == DecisionCriterion.BAYES:
        return sum(v * p for v, p in zip(values, probs))
    elif criterion == DecisionCriterion.WALD:
        return min(values)
    elif criterion == DecisionCriterion.HURWICZ:
        return alpha * max(values) + (1 - alpha) * min(values)
    elif criterion == DecisionCriterion.LAPLACE:
        return sum(values) / len(values)
    elif criterion == DecisionCriterion.SAVAGE:
        max_val = max(values)
        max_regret = max(max_val - v for v in values)
        return max_val - max_regret
    return sum(v * p for v, p in zip(values, probs))


def evaluate_trajectory(controls_seq: List[Tuple], criterion: DecisionCriterion, 
                       alpha: float = 0.5) -> float:
    """Оценить траекторию управлений по всем возможным сценариям"""
    scenario_indices = list(product(range(3), repeat=3))
    
    final_values = []
    final_probs = []
    
    for sc_combo in scenario_indices:
        state = INIT_STATE[:]
        prob = 1.0
        
        for stage, (u, sc_idx) in enumerate(zip(controls_seq, sc_combo)):
            state = apply_control(state, u)
            if any(x < -0.01 for x in state):
                prob = 0
                break
            
            scenario = STAGES_DATA[stage][sc_idx]
            prob *= scenario[0]
            state = apply_scenario(state, scenario)
        
        if prob > 0:
            final_values.append(total_value(state))
            final_probs.append(prob)
    
    if not final_values:
        return float('-inf')
    
    return evaluate_by_criterion(final_values, final_probs, criterion, alpha)


def solve(criterion: DecisionCriterion = DecisionCriterion.BAYES, 
          alpha: float = 0.5) -> Tuple[float, List[Tuple]]:
    """Решить задачу полным перебором траекторий"""
    controls_0 = get_controls(INIT_STATE)
    
    best_value = float('-inf')
    best_trajectory = [(0, 0, 0)] * 3
    
    for u0 in controls_0:
        state1 = apply_control(INIT_STATE, u0)
        if any(x < 0 for x in state1):
            continue
        
        # Ожидаемое состояние после этапа 0
        exp_state1 = [0, 0, 0, state1[3]]
        for sc in STAGES_DATA[0]:
            s = apply_scenario(state1, sc)
            exp_state1[0] += s[0] * sc[0]
            exp_state1[1] += s[1] * sc[0]
            exp_state1[2] += s[2] * sc[0]
        
        controls_1 = get_controls(exp_state1)
        
        for u1 in controls_1:
            state2 = apply_control(exp_state1, u1)
            if any(x < 0 for x in state2):
                continue
            
            exp_state2 = [0, 0, 0, state2[3]]
            for sc in STAGES_DATA[1]:
                s = apply_scenario(state2, sc)
                exp_state2[0] += s[0] * sc[0]
                exp_state2[1] += s[1] * sc[0]
                exp_state2[2] += s[2] * sc[0]
            
            controls_2 = get_controls(exp_state2)
            
            for u2 in controls_2:
                state3 = apply_control(exp_state2, u2)
                if any(x < 0 for x in state3):
                    continue
                
                trajectory = [u0, u1, u2]
                value = evaluate_trajectory(trajectory, criterion, alpha)
                
                if value > best_value:
                    best_value = value
                    best_trajectory = trajectory
    
    return best_value, best_trajectory


def control_to_string(u: Tuple[int, int, int]) -> str:
    """Преобразовать управление в читаемую строку"""
    parts = []
    labels = ['ЦБ1', 'ЦБ2', 'Деп.']
    for i, ui in enumerate(u):
        if ui != 0:
            act = 'купить' if ui > 0 else 'продать'
            parts.append(f"{act} {abs(ui)} пак. {labels[i]} ({abs(ui)*DELTA[i]} д.е.)")
    return ', '.join(parts) if parts else 'ничего не делать'


def print_problem_data():
    """Вывести исходные данные задачи"""
    print("\n" + "=" * 70)
    print("ИСХОДНЫЕ ДАННЫЕ ЗАДАЧИ")
    print("=" * 70)
    
    print("\nНачальное состояние портфеля:")
    print(f"  ЦБ1: {INIT_STATE[0]:.0f} д.е. (пакет = {DELTA[0]} д.е.)")
    print(f"  ЦБ2: {INIT_STATE[1]:.0f} д.е. (пакет = {DELTA[1]} д.е.)")
    print(f"  Депозит: {INIT_STATE[2]:.0f} д.е. (пакет = {DELTA[2]} д.е.)")
    print(f"  Свободные средства: {INIT_STATE[3]:.0f} д.е.")
    print(f"  ИТОГО: {total_value(INIT_STATE):.0f} д.е.")
    
    print("\nКоэффициенты изменения стоимости:")
    for i, stage in enumerate(STAGES_DATA):
        print(f"\n  Этап {i+1}:")
        print(f"  {'Ситуация':<15} {'p':<6} {'ЦБ1':<6} {'ЦБ2':<6} {'Деп.':<6}")
        for j, (p, r1, r2, r3) in enumerate(stage):
            print(f"  {SCENARIO_NAMES[j]:<15} {p:<6.2f} {r1:<6.2f} {r2:<6.2f} {r3:<6.2f}")


def print_detailed_solution(criterion: DecisionCriterion, alpha: float = 0.5):
    """Вывести подробное решение"""
    print("\n" + "=" * 70)
    print(f"РЕШЕНИЕ ПО КРИТЕРИЮ: {criterion.value}")
    if criterion == DecisionCriterion.HURWICZ:
        print(f"Коэффициент оптимизма α = {alpha}")
    print("=" * 70)
    
    opt_val, trajectory = solve(criterion, alpha)
    
    # Симуляция траектории
    state = INIT_STATE[:]
    
    for stage in range(3):
        u = trajectory[stage]
        print(f"\n--- ЭТАП {stage + 1} ---")
        print(f"Состояние: ЦБ1={state[0]:.2f}, ЦБ2={state[1]:.2f}, "
              f"Деп={state[2]:.2f}, Своб.ср.={state[3]:.2f}")
        print(f"Стоимость портфеля: {total_value(state):.2f} д.е.")
        print(f"Управление: {control_to_string(u)}")
        
        # Применяем управление
        state = apply_control(state, u)
        print(f"После управления: ЦБ1={state[0]:.2f}, ЦБ2={state[1]:.2f}, "
              f"Деп={state[2]:.2f}, Своб.ср.={state[3]:.2f}")
        
        # Возможные исходы
        print("Возможные исходы:")
        for j, sc in enumerate(STAGES_DATA[stage]):
            next_s = apply_scenario(state, sc)
            print(f"  - {SCENARIO_NAMES[j]} (p={sc[0]:.0%}): {total_value(next_s):.2f} д.е.")
        
        # Ожидаемое состояние
        exp_state = [0, 0, 0, state[3]]
        for sc in STAGES_DATA[stage]:
            s = apply_scenario(state, sc)
            exp_state[0] += s[0] * sc[0]
            exp_state[1] += s[1] * sc[0]
            exp_state[2] += s[2] * sc[0]
        state = exp_state
    
    initial = total_value(INIT_STATE)
    profit = opt_val - initial
    returns = (opt_val / initial - 1) * 100
    
    print("\n" + "=" * 70)
    print("ИТОГ:")
    print(f"  Начальный капитал: {initial:.2f} д.е.")
    print(f"  Ожидаемый капитал: {opt_val:.2f} д.е.")
    print(f"  Доход: {profit:.2f} д.е. ({returns:.2f}%)")
    print("=" * 70)
    
    return opt_val, trajectory


def compare_all_criteria(alpha: float = 0.5):
    """Сравнить все критерии"""
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ ВСЕХ КРИТЕРИЕВ")
    print("=" * 70)
    
    results = {}
    trajectories = {}
    initial = total_value(INIT_STATE)
    
    for criterion in DecisionCriterion:
        opt_val, traj = solve(criterion, alpha)
        results[criterion] = opt_val
        trajectories[criterion] = traj
        
        profit = opt_val - initial
        returns = (opt_val / initial - 1) * 100
        
        print(f"\nКритерий {criterion.value}:")
        print(f"  Ожидаемый капитал: {opt_val:.2f} д.е.")
        print(f"  Доход: {profit:.2f} д.е. ({returns:.2f}%)")
        print(f"  Стратегия:")
        for i, u in enumerate(traj):
            print(f"    Этап {i+1}: {control_to_string(u)}")
    
    # Лучший результат
    best = max(results.items(), key=lambda x: x[1])
    print("\n" + "=" * 70)
    print(f"ЛУЧШИЙ РЕЗУЛЬТАТ: Критерий {best[0].value}")
    print(f"Ожидаемый капитал: {best[1]:.2f} д.е.")
    print("=" * 70)


def main():
    """Главная функция с интерактивным меню"""
    print("\n" + "=" * 70)
    print("ПРОГРАММА ОПТИМИЗАЦИИ ИНВЕСТИЦИОННОГО ПОРТФЕЛЯ")
    print("Метод динамического программирования")
    print("=" * 70)
    
    while True:
        print("\n" + "-" * 40)
        print("МЕНЮ")
        print("-" * 40)
        print("1. Критерий Байеса (математическое ожидание)")
        print("2. Критерий Вальда (максимин)")
        print("3. Критерий Сэвиджа (минимакс сожалений)")
        print("4. Критерий Гурвица (оптимизм-пессимизм)")
        print("5. Критерий Лапласа (равновероятность)")
        print("6. Сравнить все критерии")
        print("7. Показать исходные данные")
        print("0. Выход")
        print("-" * 40)
        
        try:
            choice = input("Выберите (0-7): ").strip()
        except EOFError:
            choice = "1"
        
        if choice == "0":
            print("\nДо свидания!")
            break
        elif choice == "1":
            print_detailed_solution(DecisionCriterion.BAYES)
        elif choice == "2":
            print_detailed_solution(DecisionCriterion.WALD)
        elif choice == "3":
            print_detailed_solution(DecisionCriterion.SAVAGE)
        elif choice == "4":
            try:
                alpha_str = input("Введите α (0-1) [0.5]: ").strip()
                alpha = float(alpha_str) if alpha_str else 0.5
                alpha = max(0, min(1, alpha))
            except (ValueError, EOFError):
                alpha = 0.5
            print_detailed_solution(DecisionCriterion.HURWICZ, alpha)
        elif choice == "5":
            print_detailed_solution(DecisionCriterion.LAPLACE)
        elif choice == "6":
            try:
                alpha_str = input("Введите α для Гурвица (0-1) [0.5]: ").strip()
                alpha = float(alpha_str) if alpha_str else 0.5
            except (ValueError, EOFError):
                alpha = 0.5
            compare_all_criteria(alpha)
        elif choice == "7":
            print_problem_data()
        else:
            print("Неверный выбор!")
        
        try:
            cont = input("\nПродолжить? (y/n) [y]: ").strip().lower()
            if cont == 'n':
                break
        except EOFError:
            break


if __name__ == "__main__":
    main()
