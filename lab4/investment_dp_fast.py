# -*- coding: utf-8 -*-
"""
Оптимизация инвестиционного портфеля - быстрая версия.
Полный перебор траекторий с ограниченным набором управлений.
"""

from enum import Enum
from itertools import product


class Criterion(Enum):
    BAYES = "Байеса (мат. ожидание)"
    WALD = "Вальда (максимин)"
    HURWICZ = "Гурвица (α=0.5)"
    LAPLACE = "Лапласа (равновероятность)"


# Глобальные параметры
INIT = [100.0, 800.0, 400.0, 600.0]  # ЦБ1, ЦБ2, Деп, Своб.ср.
D = [25, 200, 100]  # Шаги: ЦБ1, ЦБ2, Деп

# Коэффициенты: (prob, r_цб1, r_цб2, r_деп)
STAGES = [
    [(0.60, 1.20, 1.10, 1.07), (0.30, 1.05, 1.02, 1.03), (0.10, 0.80, 0.95, 1.00)],
    [(0.30, 1.40, 1.15, 1.01), (0.20, 1.05, 1.00, 1.00), (0.50, 0.60, 0.90, 1.00)],
    [(0.40, 1.15, 1.12, 1.05), (0.40, 1.05, 1.01, 1.01), (0.20, 0.70, 0.94, 1.00)],
]


def get_smart_controls(state):
    """Генерация умного набора управлений"""
    cb1, cb2, dep, cash = state
    controls = []
    
    for u1 in range(-1, 2):
        for u2 in range(-1, 2):
            for u3 in range(-1, 2):
                new_cb1 = cb1 + u1 * D[0]
                new_cb2 = cb2 + u2 * D[1]
                new_dep = dep + u3 * D[2]
                cost = u1 * D[0] + u2 * D[1] + u3 * D[2]
                new_cash = cash - cost
                
                if new_cb1 >= 0 and new_cb2 >= 0 and new_dep >= 0 and new_cash >= 0:
                    controls.append((u1, u2, u3))
    
    # Агрессивные варианты
    max_u1 = int(cash // D[0])
    if max_u1 > 1:
        controls.append((min(max_u1, 4), 0, 0))
    
    max_u2 = int(cash // D[1])
    if max_u2 > 1:
        controls.append((0, min(max_u2, 2), 0))
    
    max_u3 = int(cash // D[2])
    if max_u3 > 1:
        controls.append((0, 0, min(max_u3, 3)))
    
    return list(set(controls))


def apply_control(state, u):
    """Применить управление"""
    cost = u[0] * D[0] + u[1] * D[1] + u[2] * D[2]
    return [state[0] + u[0] * D[0], state[1] + u[1] * D[1], 
            state[2] + u[2] * D[2], state[3] - cost]


def apply_scenario(state, scenario):
    """Применить сценарий"""
    _, r1, r2, r3 = scenario
    return [state[0] * r1, state[1] * r2, state[2] * r3, state[3]]


def total(state):
    return sum(state)


def evaluate(values, probs, crit, alpha=0.5):
    """Оценить по критерию"""
    if crit == Criterion.BAYES:
        return sum(v * p for v, p in zip(values, probs))
    elif crit == Criterion.WALD:
        return min(values)
    elif crit == Criterion.HURWICZ:
        return alpha * max(values) + (1 - alpha) * min(values)
    elif crit == Criterion.LAPLACE:
        return sum(values) / len(values)
    return sum(v * p for v, p in zip(values, probs))


def evaluate_trajectory(controls_seq, criterion, alpha=0.5):
    """Оценить траекторию по всем сценариям"""
    scenario_indices = list(product(range(3), repeat=3))
    
    final_values = []
    final_probs = []
    
    for sc_combo in scenario_indices:
        state = INIT[:]
        prob = 1.0
        
        for stage, (u, sc_idx) in enumerate(zip(controls_seq, sc_combo)):
            state = apply_control(state, u)
            if any(x < -0.01 for x in state):
                prob = 0
                break
            
            scenario = STAGES[stage][sc_idx]
            prob *= scenario[0]
            state = apply_scenario(state, scenario)
        
        if prob > 0:
            final_values.append(total(state))
            final_probs.append(prob)
    
    if not final_values:
        return float('-inf')
    
    return evaluate(final_values, final_probs, criterion, alpha)


def solve(criterion=Criterion.BAYES, alpha=0.5):
    """Решить задачу полным перебором траекторий"""
    controls_0 = get_smart_controls(INIT)
    
    best_value = float('-inf')
    best_trajectory = [(0, 0, 0)] * 3
    total_checked = 0
    
    for u0 in controls_0:
        state1 = apply_control(INIT, u0)
        if any(x < 0 for x in state1):
            continue
        
        exp_state1 = [0, 0, 0, state1[3]]
        for sc in STAGES[0]:
            s = apply_scenario(state1, sc)
            exp_state1[0] += s[0] * sc[0]
            exp_state1[1] += s[1] * sc[0]
            exp_state1[2] += s[2] * sc[0]
        
        controls_1 = get_smart_controls(exp_state1)
        
        for u1 in controls_1:
            state2 = apply_control(exp_state1, u1)
            if any(x < 0 for x in state2):
                continue
            
            exp_state2 = [0, 0, 0, state2[3]]
            for sc in STAGES[1]:
                s = apply_scenario(state2, sc)
                exp_state2[0] += s[0] * sc[0]
                exp_state2[1] += s[1] * sc[0]
                exp_state2[2] += s[2] * sc[0]
            
            controls_2 = get_smart_controls(exp_state2)
            
            for u2 in controls_2:
                state3 = apply_control(exp_state2, u2)
                if any(x < 0 for x in state3):
                    continue
                
                trajectory = [u0, u1, u2]
                value = evaluate_trajectory(trajectory, criterion, alpha)
                total_checked += 1
                
                if value > best_value:
                    best_value = value
                    best_trajectory = trajectory
    
    return best_value, best_trajectory, total_checked


def control_str(u):
    """Преобразовать управление в строку"""
    parts = []
    labels = ['ЦБ1', 'ЦБ2', 'Деп.']
    for i, ui in enumerate(u):
        if ui != 0:
            act = 'купить' if ui > 0 else 'продать'
            parts.append(f"{act} {abs(ui)} пак. {labels[i]} ({abs(ui)*D[i]} д.е.)")
    return ', '.join(parts) if parts else 'ничего не делать'


def print_solution(criterion, alpha=0.5):
    """Вывести решение"""
    print(f"\n{'='*70}", flush=True)
    print(f"РЕШЕНИЕ: Критерий {criterion.value}", flush=True)
    print(f"{'='*70}", flush=True)
    
    opt_val, trajectory, checked = solve(criterion, alpha)
    
    print(f"Проверено траекторий: {checked}", flush=True)
    
    state = INIT[:]
    
    for stage in range(3):
        u = trajectory[stage]
        print(f"\n--- ЭТАП {stage + 1} ---", flush=True)
        print(f"Состояние: ЦБ1={state[0]:.2f}, ЦБ2={state[1]:.2f}, "
              f"Деп={state[2]:.2f}, Своб.ср.={state[3]:.2f}", flush=True)
        print(f"Стоимость: {total(state):.2f} д.е.", flush=True)
        print(f"Управление: {control_str(u)}", flush=True)
        
        state = apply_control(state, u)
        print(f"После управления: ЦБ1={state[0]:.2f}, ЦБ2={state[1]:.2f}, "
              f"Деп={state[2]:.2f}, Своб.ср.={state[3]:.2f}", flush=True)
        
        print("Возможные исходы:", flush=True)
        names = ["Благопр.", "Нейтр.", "Негатив."]
        for i, sc in enumerate(STAGES[stage]):
            next_s = apply_scenario(state, sc)
            print(f"  - {names[i]} (p={sc[0]:.0%}): {total(next_s):.2f} д.е.", flush=True)
        
        exp_state = [0, 0, 0, state[3]]
        for sc in STAGES[stage]:
            s = apply_scenario(state, sc)
            exp_state[0] += s[0] * sc[0]
            exp_state[1] += s[1] * sc[0]
            exp_state[2] += s[2] * sc[0]
        state = exp_state
    
    profit = opt_val - total(INIT)
    returns = (opt_val / total(INIT) - 1) * 100
    
    print(f"\n{'='*70}", flush=True)
    print("ИТОГ:", flush=True)
    print(f"  Начальный капитал: {total(INIT):.2f} д.е.", flush=True)
    print(f"  Ожидаемый капитал: {opt_val:.2f} д.е.", flush=True)
    print(f"  Доход: {profit:.2f} д.е. ({returns:.2f}%)", flush=True)
    
    return opt_val, trajectory


def main():
    print("=" * 70, flush=True)
    print("ОПТИМИЗАЦИЯ ИНВЕСТИЦИОННОГО ПОРТФЕЛЯ", flush=True)
    print("Метод динамического программирования", flush=True)
    print("=" * 70, flush=True)
    
    print("\nИСХОДНЫЕ ДАННЫЕ:", flush=True)
    print(f"  ЦБ1: {INIT[0]:.0f} д.е. (пакет = {D[0]} д.е.)", flush=True)
    print(f"  ЦБ2: {INIT[1]:.0f} д.е. (пакет = {D[1]} д.е.)", flush=True)
    print(f"  Депозит: {INIT[2]:.0f} д.е. (пакет = {D[2]} д.е.)", flush=True)
    print(f"  Свободные средства: {INIT[3]:.0f} д.е.", flush=True)
    print(f"  ИТОГО: {total(INIT):.0f} д.е.", flush=True)
    
    print("\nКОЭФФИЦИЕНТЫ ИЗМЕНЕНИЯ СТОИМОСТИ:", flush=True)
    names = ["Благопр.", "Нейтр.", "Негатив."]
    for i, stage in enumerate(STAGES):
        print(f"\n  Этап {i+1}:", flush=True)
        print(f"  {'Ситуация':<10} {'p':<6} {'ЦБ1':<6} {'ЦБ2':<6} {'Деп.':<6}", flush=True)
        for j, (p, r1, r2, r3) in enumerate(stage):
            print(f"  {names[j]:<10} {p:<6.2f} {r1:<6.2f} {r2:<6.2f} {r3:<6.2f}", flush=True)
    
    # Решение по всем критериям
    results = {}
    trajectories = {}
    
    for crit in Criterion:
        opt_val, traj = print_solution(crit)
        results[crit] = opt_val
        trajectories[crit] = traj
    
    # Сравнение
    print(f"\n{'='*70}", flush=True)
    print("СРАВНЕНИЕ КРИТЕРИЕВ:", flush=True)
    print(f"{'='*70}", flush=True)
    
    for crit, val in results.items():
        profit = val - total(INIT)
        traj = trajectories[crit]
        print(f"\nКритерий {crit.value}:", flush=True)
        print(f"  Капитал: {val:.2f} д.е., Доход: {profit:.2f} д.е.", flush=True)
        print(f"  Этап 1: {control_str(traj[0])}", flush=True)
        print(f"  Этап 2: {control_str(traj[1])}", flush=True)
        print(f"  Этап 3: {control_str(traj[2])}", flush=True)
    
    best = max(results.items(), key=lambda x: x[1])
    print(f"\n{'='*70}", flush=True)
    print(f"ЛУЧШИЙ РЕЗУЛЬТАТ: Критерий {best[0].value}", flush=True)
    print(f"Ожидаемый капитал: {best[1]:.2f} д.е.", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
