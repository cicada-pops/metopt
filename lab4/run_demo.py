# -*- coding: utf-8 -*-
"""
Демонстрационный скрипт для запуска оптимизации портфеля.
"""

import sys

# Принудительно отключаем буферизацию
sys.stdout.flush()

from investment_optimizer import InvestmentOptimizer, DecisionCriterion

def main():
    optimizer = InvestmentOptimizer()
    
    print("=" * 80, flush=True)
    print("OPTIMIZATSIYA INVESTITSIONNOGO PORTFELYA", flush=True)
    print("Metod dinamicheskogo programmirovaniya", flush=True)
    print("=" * 80, flush=True)
    
    # Показываем исходные данные
    print("\n" + "=" * 80, flush=True)
    print("ISHODNYIE DANNYIE ZADACHI", flush=True)
    print("=" * 80, flush=True)
    
    print("\nNachalnoe sostoyanie:", flush=True)
    print(f"  CB1: {optimizer.initial_state.cb1} d.e.", flush=True)
    print(f"  CB2: {optimizer.initial_state.cb2} d.e.", flush=True)
    print(f"  Depozit: {optimizer.initial_state.dep} d.e.", flush=True)
    print(f"  Svobodnyie sredstva: {optimizer.initial_state.cash} d.e.", flush=True)
    print(f"  ITOGO: {optimizer.initial_state.total_value()} d.e.", flush=True)
    
    print("\nRazmery paketov (1/4 ot nachalnoy stoimosti):", flush=True)
    print(f"  CB1: {optimizer.delta_cb1} d.e.", flush=True)
    print(f"  CB2: {optimizer.delta_cb2} d.e.", flush=True)
    print(f"  Depozit: {optimizer.delta_dep} d.e.", flush=True)
    
    print("\nKoeffitsienty izmeneniya stoimosti:", flush=True)
    for stage_idx, stage_data in enumerate(optimizer.stages_data):
        print(f"\n  Etap {stage_idx + 1}:", flush=True)
        print(f"  {'Situatsiya':<15} {'Veroyat.':<10} {'CB1':<8} {'CB2':<8} {'Dep.':<8}", flush=True)
        print(f"  {'-' * 49}", flush=True)
        for scenario in stage_data:
            print(f"  {scenario.name:<15} {scenario.probability:<10.2f} "
                  f"{scenario.r_cb1:<8.2f} {scenario.r_cb2:<8.2f} {scenario.r_dep:<8.2f}", flush=True)
    
    # Решаем по критерию Байеса
    print("\n" + "=" * 80, flush=True)
    print("RESHENIE PO KRITERIYU BAYESA (matematicheskoe ozhidanie)", flush=True)
    print("=" * 80, flush=True)
    
    optimal_value, trajectory = optimizer.solve(DecisionCriterion.BAYES)
    
    print("\nOptimalnaya strategiya:", flush=True)
    for stage, (state, control) in enumerate(trajectory):
        print(f"\n--- ETAP {stage + 1} ---", flush=True)
        print(f"Sostoyanie: CB1={state.cb1:.2f}, CB2={state.cb2:.2f}, "
              f"Dep={state.dep:.2f}, Cash={state.cash:.2f}", flush=True)
        print(f"Obshchaya stoimost: {state.total_value():.2f} d.e.", flush=True)
        print(f"Upravlenie: {control}", flush=True)
        
        # Промежуточное состояние
        intermediate = optimizer.apply_control(state, control)
        print(f"Posle upravleniya: CB1={intermediate.cb1:.2f}, CB2={intermediate.cb2:.2f}, "
              f"Dep={intermediate.dep:.2f}, Cash={intermediate.cash:.2f}", flush=True)
        
        # Возможные исходы
        print("Vozmozhnye iskhody:", flush=True)
        scenarios = optimizer.stages_data[stage]
        for scenario in scenarios:
            next_state = optimizer.apply_scenario(intermediate, scenario)
            print(f"  - {scenario.name} (p={scenario.probability:.0%}): "
                  f"Stoimost = {next_state.total_value():.2f} d.e.", flush=True)
    
    initial_value = optimizer.initial_state.total_value()
    profit = optimal_value - initial_value
    returns = (optimal_value / initial_value - 1) * 100
    
    print("\n" + "=" * 80, flush=True)
    print("ITOGOVYE REZULTATY (Kriteriy Bayesa)", flush=True)
    print("=" * 80, flush=True)
    print(f"Nachalnyy kapital: {initial_value:.2f} d.e.", flush=True)
    print(f"Ozhidaemyy itogovyy kapital: {optimal_value:.2f} d.e.", flush=True)
    print(f"Ozhidaemyy dokhod: {profit:.2f} d.e.", flush=True)
    print(f"Ozhidaemaya dokhodnost: {returns:.2f}%", flush=True)
    
    # Сравнение критериев
    print("\n" + "=" * 80, flush=True)
    print("SRAVNENIE KRITERIEV PRINYATIYA RESHENIY", flush=True)
    print("=" * 80, flush=True)
    
    criteria_names = {
        DecisionCriterion.BAYES: "Bayes (mat. ozhidanie)",
        DecisionCriterion.WALD: "Wald (maksimin)",
        DecisionCriterion.SAVAGE: "Savage (minimaks sozhaleniy)",
        DecisionCriterion.HURWICZ: "Hurwicz (alpha=0.5)",
        DecisionCriterion.LAPLACE: "Laplace (ravnoveroyatnost)",
    }
    
    results = {}
    for criterion in DecisionCriterion:
        optimal_value, trajectory = optimizer.solve(criterion, hurwicz_alpha=0.5)
        profit = optimal_value - initial_value
        returns = (optimal_value / initial_value - 1) * 100
        results[criterion] = optimal_value
        
        print(f"\n{criteria_names[criterion]}:", flush=True)
        print(f"  Ozhidaemyy kapital: {optimal_value:.2f} d.e.", flush=True)
        print(f"  Dokhod: {profit:.2f} d.e. ({returns:.2f}%)", flush=True)
        print(f"  Strategiya etap 1: {trajectory[0][1]}", flush=True)
    
    # Лучший результат
    best_criterion = max(results.items(), key=lambda x: x[1])
    print("\n" + "=" * 80, flush=True)
    print(f"LUCHSHIY REZULTAT: {criteria_names[best_criterion[0]]}", flush=True)
    print(f"Ozhidaemyy kapital: {best_criterion[1]:.2f} d.e.", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
