# -*- coding: utf-8 -*-
"""
Оптимизация инвестиционного портфеля методом динамического программирования.
Оптимизированная версия с ограничением пространства управлений.
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import itertools


class DecisionCriterion(Enum):
    """Критерии принятия решений"""
    BAYES = "Bayes (matematicheskoe ozhidanie)"
    WALD = "Wald (maksimin)"
    SAVAGE = "Savage (minimaks sozhaleniy)"
    HURWICZ = "Hurwicz (optimizm-pessimizm)"
    LAPLACE = "Laplace (ravnoveroyatnost)"


@dataclass
class State:
    """Состояние портфеля"""
    cb1: float
    cb2: float
    dep: float
    cash: float
    
    def total(self) -> float:
        return self.cb1 + self.cb2 + self.dep + self.cash
    
    def copy(self):
        return State(self.cb1, self.cb2, self.dep, self.cash)


@dataclass
class Scenario:
    """Сценарий (ситуация)"""
    name: str
    prob: float
    r1: float  # Коэффициент ЦБ1
    r2: float  # Коэффициент ЦБ2
    r3: float  # Коэффициент депозита


class PortfolioOptimizer:
    """Оптимизатор портфеля методом ДП"""
    
    def __init__(self):
        # Начальное состояние
        self.init_state = State(cb1=100, cb2=800, dep=400, cash=600)
        
        # Шаги управления
        self.d1 = 25   # ЦБ1
        self.d2 = 200  # ЦБ2
        self.d3 = 100  # Депозит
        
        # Данные по этапам
        self.stages = [
            [  # Этап 1
                Scenario("Blagopr.", 0.60, 1.20, 1.10, 1.07),
                Scenario("Neytr.", 0.30, 1.05, 1.02, 1.03),
                Scenario("Negativ.", 0.10, 0.80, 0.95, 1.00),
            ],
            [  # Этап 2
                Scenario("Blagopr.", 0.30, 1.40, 1.15, 1.01),
                Scenario("Neytr.", 0.20, 1.05, 1.00, 1.00),
                Scenario("Negativ.", 0.50, 0.60, 0.90, 1.00),
            ],
            [  # Этап 3
                Scenario("Blagopr.", 0.40, 1.15, 1.12, 1.05),
                Scenario("Neytr.", 0.40, 1.05, 1.01, 1.01),
                Scenario("Negativ.", 0.20, 0.70, 0.94, 1.00),
            ],
        ]
        
        self.hurwicz_alpha = 0.5
    
    def get_controls(self, state: State) -> List[Tuple[int, int, int]]:
        """Генерация допустимых управлений"""
        controls = []
        
        # Ограничиваем диапазон для ускорения
        u1_min = max(-4, -int(state.cb1 // self.d1))
        u1_max = min(24, int(state.cash // self.d1))
        
        u2_min = max(-4, -int(state.cb2 // self.d2))
        u2_max = min(3, int(state.cash // self.d2))
        
        u3_min = max(-4, -int(state.dep // self.d3))
        u3_max = min(6, int(state.cash // self.d3))
        
        for u1 in range(u1_min, u1_max + 1):
            for u2 in range(u2_min, u2_max + 1):
                for u3 in range(u3_min, u3_max + 1):
                    cost = u1 * self.d1 + u2 * self.d2 + u3 * self.d3
                    if cost <= state.cash + 0.01:
                        # Проверяем, что активы не отрицательные
                        new_cb1 = state.cb1 + u1 * self.d1
                        new_cb2 = state.cb2 + u2 * self.d2
                        new_dep = state.dep + u3 * self.d3
                        if new_cb1 >= -0.01 and new_cb2 >= -0.01 and new_dep >= -0.01:
                            controls.append((u1, u2, u3))
        
        return controls
    
    def apply_control(self, state: State, u: Tuple[int, int, int]) -> State:
        """Применить управление"""
        cost = u[0] * self.d1 + u[1] * self.d2 + u[2] * self.d3
        return State(
            cb1=state.cb1 + u[0] * self.d1,
            cb2=state.cb2 + u[1] * self.d2,
            dep=state.dep + u[2] * self.d3,
            cash=state.cash - cost
        )
    
    def apply_scenario(self, state: State, sc: Scenario) -> State:
        """Применить сценарий"""
        return State(
            cb1=state.cb1 * sc.r1,
            cb2=state.cb2 * sc.r2,
            dep=state.dep * sc.r3,
            cash=state.cash
        )
    
    def evaluate(self, values: List[float], probs: List[float], criterion: DecisionCriterion) -> float:
        """Оценить по критерию"""
        if criterion == DecisionCriterion.BAYES:
            return sum(v * p for v, p in zip(values, probs))
        elif criterion == DecisionCriterion.WALD:
            return min(values)
        elif criterion == DecisionCriterion.HURWICZ:
            return self.hurwicz_alpha * max(values) + (1 - self.hurwicz_alpha) * min(values)
        elif criterion == DecisionCriterion.LAPLACE:
            return sum(values) / len(values)
        elif criterion == DecisionCriterion.SAVAGE:
            max_val = max(values)
            max_regret = max(max_val - v for v in values)
            return max_val - max_regret
        return sum(v * p for v, p in zip(values, probs))
    
    def solve_recursive(self, state: State, stage: int, criterion: DecisionCriterion,
                       memo: Dict) -> Tuple[float, List[Tuple[int, int, int]]]:
        """
        Рекурсивное решение с мемоизацией.
        Возвращает (оптимальное значение, список управлений).
        """
        # Терминальное условие
        if stage >= 3:
            return state.total(), []
        
        # Ключ для мемоизации (округляем для избежания проблем с float)
        key = (round(state.cb1, 1), round(state.cb2, 1), 
               round(state.dep, 1), round(state.cash, 1), stage)
        
        if key in memo:
            return memo[key]
        
        controls = self.get_controls(state)
        scenarios = self.stages[stage]
        
        best_value = float('-inf')
        best_control = (0, 0, 0)
        best_path = []
        
        for control in controls:
            # Применяем управление
            after_control = self.apply_control(state, control)
            
            # Вычисляем значения для каждого сценария
            scenario_values = []
            scenario_paths = []
            
            for sc in scenarios:
                next_state = self.apply_scenario(after_control, sc)
                val, path = self.solve_recursive(next_state, stage + 1, criterion, memo)
                scenario_values.append(val)
                scenario_paths.append(path)
            
            # Оцениваем по критерию
            probs = [sc.prob for sc in scenarios]
            eval_value = self.evaluate(scenario_values, probs, criterion)
            
            if eval_value > best_value:
                best_value = eval_value
                best_control = control
                # Для пути берём наиболее вероятный сценарий
                max_prob_idx = probs.index(max(probs))
                best_path = scenario_paths[max_prob_idx]
        
        result = (best_value, [best_control] + best_path)
        memo[key] = result
        return result
    
    def solve(self, criterion: DecisionCriterion = DecisionCriterion.BAYES,
              hurwicz_alpha: float = 0.5) -> Tuple[float, List[Tuple[int, int, int]]]:
        """Решить задачу"""
        self.hurwicz_alpha = hurwicz_alpha
        memo = {}
        return self.solve_recursive(self.init_state, 0, criterion, memo)
    
    def control_to_str(self, u: Tuple[int, int, int]) -> str:
        """Преобразовать управление в строку"""
        actions = []
        if u[0] != 0:
            act = "kupit" if u[0] > 0 else "prodat"
            actions.append(f"{act} {abs(u[0])} pak. CB1 ({abs(u[0]) * self.d1} d.e.)")
        if u[1] != 0:
            act = "kupit" if u[1] > 0 else "prodat"
            actions.append(f"{act} {abs(u[1])} pak. CB2 ({abs(u[1]) * self.d2} d.e.)")
        if u[2] != 0:
            act = "kupit" if u[2] > 0 else "prodat"
            actions.append(f"{act} {abs(u[2])} pak. Dep. ({abs(u[2]) * self.d3} d.e.)")
        return ", ".join(actions) if actions else "nichego ne delat"


def print_solution(opt: PortfolioOptimizer, criterion: DecisionCriterion, alpha: float = 0.5):
    """Вывести решение"""
    print(f"\n{'='*70}", flush=True)
    print(f"RESHENIE PO KRITERIYU: {criterion.value}", flush=True)
    if criterion == DecisionCriterion.HURWICZ:
        print(f"Koeffitsient optimizma alpha = {alpha}", flush=True)
    print(f"{'='*70}", flush=True)
    
    opt_value, controls = opt.solve(criterion, alpha)
    
    # Симуляция траектории
    state = opt.init_state.copy()
    
    for stage in range(3):
        print(f"\n--- ETAP {stage + 1} ---", flush=True)
        print(f"Sostoyanie: CB1={state.cb1:.2f}, CB2={state.cb2:.2f}, "
              f"Dep={state.dep:.2f}, Cash={state.cash:.2f}", flush=True)
        print(f"Obshchaya stoimost: {state.total():.2f} d.e.", flush=True)
        
        if stage < len(controls):
            u = controls[stage]
            print(f"Upravlenie: {opt.control_to_str(u)}", flush=True)
            
            # Применяем управление
            state = opt.apply_control(state, u)
            print(f"Posle upravleniya: CB1={state.cb1:.2f}, CB2={state.cb2:.2f}, "
                  f"Dep={state.dep:.2f}, Cash={state.cash:.2f}", flush=True)
            
            # Показываем исходы
            print("Vozmozhnye iskhody:", flush=True)
            scenarios = opt.stages[stage]
            expected_cb1 = 0
            expected_cb2 = 0
            expected_dep = 0
            for sc in scenarios:
                next_s = opt.apply_scenario(state, sc)
                print(f"  - {sc.name} (p={sc.prob:.0%}): "
                      f"Stoimost = {next_s.total():.2f} d.e.", flush=True)
                expected_cb1 += next_s.cb1 * sc.prob
                expected_cb2 += next_s.cb2 * sc.prob
                expected_dep += next_s.dep * sc.prob
            
            # Обновляем состояние для следующего этапа (ожидаемое)
            state = State(expected_cb1, expected_cb2, expected_dep, state.cash)
    
    initial = opt.init_state.total()
    profit = opt_value - initial
    returns = (opt_value / initial - 1) * 100
    
    print(f"\n{'='*70}", flush=True)
    print("ITOG:", flush=True)
    print(f"  Nachalnyy kapital: {initial:.2f} d.e.", flush=True)
    print(f"  Ozhidaemyy kapital: {opt_value:.2f} d.e.", flush=True)
    print(f"  Ozhidaemyy dokhod: {profit:.2f} d.e. ({returns:.2f}%)", flush=True)
    print(f"{'='*70}", flush=True)
    
    return opt_value, controls


def main():
    """Главная функция"""
    opt = PortfolioOptimizer()
    
    print("=" * 70, flush=True)
    print("OPTIMIZATSIYA INVESTITSIONNOGO PORTFELYA", flush=True)
    print("Metod dinamicheskogo programmirovaniya", flush=True)
    print("=" * 70, flush=True)
    
    # Исходные данные
    print("\nISHODNYIE DANNYIE:", flush=True)
    print(f"  CB1: {opt.init_state.cb1} d.e. (paket = {opt.d1} d.e.)", flush=True)
    print(f"  CB2: {opt.init_state.cb2} d.e. (paket = {opt.d2} d.e.)", flush=True)
    print(f"  Depozit: {opt.init_state.dep} d.e. (paket = {opt.d3} d.e.)", flush=True)
    print(f"  Svobodnye sredstva: {opt.init_state.cash} d.e.", flush=True)
    print(f"  ITOGO: {opt.init_state.total()} d.e.", flush=True)
    
    print("\nKOEFFITSIENTY IZMENENIYA:", flush=True)
    for i, stage_scenarios in enumerate(opt.stages):
        print(f"\n  Etap {i+1}:", flush=True)
        print(f"  {'Situatsiya':<12} {'p':<6} {'CB1':<6} {'CB2':<6} {'Dep':<6}", flush=True)
        for sc in stage_scenarios:
            print(f"  {sc.name:<12} {sc.prob:<6.2f} {sc.r1:<6.2f} {sc.r2:<6.2f} {sc.r3:<6.2f}", flush=True)
    
    # Решение по критерию Байеса
    print_solution(opt, DecisionCriterion.BAYES)
    
    # Сравнение всех критериев
    print(f"\n{'='*70}", flush=True)
    print("SRAVNENIE KRITERIEV", flush=True)
    print(f"{'='*70}", flush=True)
    
    results = {}
    for crit in DecisionCriterion:
        val, ctrl = opt.solve(crit, 0.5)
        results[crit] = (val, ctrl)
        profit = val - opt.init_state.total()
        print(f"\n{crit.value}:", flush=True)
        print(f"  Kapital: {val:.2f} d.e., Dokhod: {profit:.2f} d.e.", flush=True)
        print(f"  Etap 1: {opt.control_to_str(ctrl[0])}", flush=True)
    
    # Лучший результат
    best = max(results.items(), key=lambda x: x[1][0])
    print(f"\n{'='*70}", flush=True)
    print(f"LUCHSHIY REZULTAT: {best[0].value}", flush=True)
    print(f"Ozhidaemyy kapital: {best[1][0]:.2f} d.e.", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()

