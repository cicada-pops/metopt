"""
Модуль для визуализации результатов оптимизации
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Dict
from matplotlib import rcParams

# Настройка для корректного отображения русского текста
rcParams['font.family'] = 'DejaVu Sans'


def plot_optimization_results(func: Callable[[float], float],
                               result: Dict,
                               a: float,
                               b: float,
                               title: str = "Поиск глобального минимума"):
    """
    Визуализация результатов оптимизации

    Args:
        func: Целевая функция
        result: Результаты оптимизации (словарь)
        a: Левая граница отрезка
        b: Правая граница отрезка
        title: Заголовок графика
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Построение исходной функции
    x_plot = np.linspace(a, b, 1000)
    y_plot = [func(x) for x in x_plot]

    # График 1: Функция и найденные точки
    ax1.plot(x_plot, y_plot, 'b-', linewidth=2, label='Исходная функция')

    # Отображение всех испытанных точек
    points = result['points']
    values = result['values']
    ax1.plot(points, values, 'go', markersize=6, label='Испытанные точки', alpha=0.6)

    # Отображение найденного минимума
    x_min = result['x_min']
    f_min = result['f_min']
    ax1.plot(x_min, f_min, 'r*', markersize=20, label=f'Найденный минимум', zorder=5)

    # Добавление аннотации для минимума
    ax1.annotate(f'x={x_min:.6f}\nf(x)={f_min:.6f}',
                 xy=(x_min, f_min),
                 xytext=(10, 30),
                 textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title(f'{title} - Исходная функция и испытанные точки', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # График 2: Функция и ломаная (оценочная функция)
    ax2.plot(x_plot, y_plot, 'b-', linewidth=2, label='Исходная функция')

    # Построение ломаной от оптимизатора
    # Создаем временный оптимизатор для получения ломаной
    from global_optimization import LipschitzOptimizer
    temp_opt = LipschitzOptimizer(func, a, b, 0.01)
    temp_opt.points = points
    temp_opt.values = values

    x_piecewise, y_piecewise = temp_opt.get_piecewise_function(num_points=1000)
    if len(x_piecewise) > 0:
        ax2.plot(x_piecewise, y_piecewise, 'g--', linewidth=1.5,
                 label='Оценочная функция (ломаная)', alpha=0.7)

    lipschitz_constant = result.get('lipschitz_constant') or temp_opt.estimate_lipschitz_constant()
    lipschitz_constant = max(lipschitz_constant, 1e-6)

    def plot_lower_polylines(ax, label=None):
        if len(points) < 2:
            return

        sorted_idx = np.argsort(points)
        sorted_points = np.array(points)[sorted_idx]
        sorted_values = np.array(values)[sorted_idx]

        for i in range(len(sorted_points) - 1):
            x_left, x_right = sorted_points[i], sorted_points[i + 1]
            if np.isclose(x_left, x_right):
                continue
            f_left, f_right = sorted_values[i], sorted_values[i + 1]

            x_star = 0.5 * (x_left + x_right) - (f_right - f_left) / (2 * lipschitz_constant)
            x_star = np.clip(x_star, x_left, x_right)
            y_star = f_left - lipschitz_constant * (x_star - x_left)

            label_to_draw = label if i == 0 else None
            ax.plot([x_left, x_star, x_right],
                    [f_left, y_star, f_right],
                    color='orange',
                    linestyle='-.',
                    linewidth=1.2,
                    alpha=0.8,
                    label=label_to_draw)

    plot_lower_polylines(ax1, label='Нижние ломаные')
    plot_lower_polylines(ax2)

    # Отображение испытанных точек
    ax2.plot(points, values, 'go', markersize=6, label='Испытанные точки', alpha=0.6)

    # Соединение испытанных точек линиями
    sorted_indices = np.argsort(points)
    sorted_points = [points[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    ax2.plot(sorted_points, sorted_values, 'c-', linewidth=1,
             label='Последовательность испытаний', alpha=0.4)

    # Отображение найденного минимума
    ax2.plot(x_min, f_min, 'r*', markersize=20, label='Найденный минимум', zorder=5)

    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('f(x)', fontsize=12)
    ax2.set_title(f'{title} - Оценочная функция', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Информация о результатах
    info_text = f"""Результаты оптимизации:
x_min = {x_min:.8f}
f(x_min) = {f_min:.8f}
Итераций: {result['iterations']}
Время: {result['time']:.4f} сек
Константа Липшица: {result['lipschitz_constant']:.4f}"""

    fig.text(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()


def print_results(result: Dict):
    """
    Вывод результатов оптимизации в консоль

    Args:
        result: Результаты оптимизации (словарь)
    """
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print("="*60)
    print(f"Приближенное значение аргумента (x_min): {result['x_min']:.10f}")
    print(f"Приближенное значение функции (f_min):  {result['f_min']:.10f}")
    print(f"Количество итераций:                     {result['iterations']}")
    print(f"Затраченное время:                       {result['time']:.6f} сек")
    print(f"Оценка константы Липшица:                {result['lipschitz_constant']:.6f}")
    print(f"Количество испытанных точек:             {len(result['points'])}")
    print("="*60 + "\n")
