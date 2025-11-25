"""
Модуль с тестовыми функциями для проверки алгоритма глобальной оптимизации
Все функции имеют несколько локальных минимумов на заданных отрезках
"""

import numpy as np
from typing import Tuple


def rastrigin_1d(x: float, A: float = 10.0) -> float:
    """
    Одномерная функция Растригина
    Имеет множество локальных минимумов
    Глобальный минимум: f(0) = 0

    Args:
        x: Аргумент функции
        A: Параметр функции (обычно 10)

    Returns:
        Значение функции
    """
    return A + x**2 - A * np.cos(2 * np.pi * x)


def rastrigin_range() -> Tuple[float, float]:
    """Рекомендуемый диапазон для функции Растригина"""
    return -5.12, 5.12


def ackley_1d(x: float, a: float = 20.0, b: float = 0.2, c: float = 2*np.pi) -> float:
    """
    Одномерная функция Экли (Ackley)
    Имеет множество локальных минимумов
    Глобальный минимум: f(0) = 0

    Args:
        x: Аргумент функции
        a, b, c: Параметры функции

    Returns:
        Значение функции
    """
    return -a * np.exp(-b * np.abs(x)) - np.exp(np.cos(c * x)) + a + np.e


def ackley_range() -> Tuple[float, float]:
    """Рекомендуемый диапазон для функции Экли"""
    return -5.0, 5.0


def griewank_1d(x: float) -> float:
    """
    Одномерная функция Гриванка (Griewank)
    Имеет множество локальных минимумов
    Глобальный минимум: f(0) = 0

    Args:
        x: Аргумент функции

    Returns:
        Значение функции
    """
    return 1 + x**2 / 4000 - np.cos(x)


def griewank_range() -> Tuple[float, float]:
    """Рекомендуемый диапазон для функции Гриванка"""
    return -10.0, 10.0


def schwefel_1d(x: float) -> float:
    """
    Модифицированная одномерная функция Швефеля (Schwefel)
    Имеет несколько локальных минимумов
    Глобальный минимум около x ≈ 420.9687

    Args:
        x: Аргумент функции

    Returns:
        Значение функции
    """
    return 418.9829 - x * np.sin(np.sqrt(np.abs(x)))


def schwefel_range() -> Tuple[float, float]:
    """Рекомендуемый диапазон для функции Швефеля"""
    return -500.0, 500.0


def multimodal_1(x: float) -> float:
    """
    Многоэкстремальная функция 1
    f(x) = x + sin(3.14159*x)
    Несколько локальных минимумов

    Args:
        x: Аргумент функции

    Returns:
        Значение функции
    """
    return x + np.sin(3.14159 * x)


def multimodal_1_range() -> Tuple[float, float]:
    """Рекомендуемый диапазон для многоэкстремальной функции 1"""
    return -10.0, 10.0


def multimodal_2(x: float) -> float:
    """
    Многоэкстремальная функция 2
    f(x) = sin(x) + sin(10x/3)
    Множество локальных минимумов

    Args:
        x: Аргумент функции

    Returns:
        Значение функции
    """
    return np.sin(x) + np.sin(10 * x / 3)


def multimodal_2_range() -> Tuple[float, float]:
    """Рекомендуемый диапазон для многоэкстремальной функции 2"""
    return 2.7, 7.5


def shubert_1d(x: float) -> float:
    """
    Одномерная функция Шуберта (Shubert)
    Имеет множество локальных минимумов

    Args:
        x: Аргумент функции

    Returns:
        Значение функции
    """
    sum_val = 0
    for i in range(1, 6):
        sum_val += i * np.cos((i + 1) * x + i)
    return sum_val


def shubert_range() -> Tuple[float, float]:
    """Рекомендуемый диапазон для функции Шуберта"""
    return -10.0, 10.0


def levy_1d(x: float) -> float:
    """
    Одномерная функция Леви (Levy)
    Имеет несколько локальных минимумов
    Глобальный минимум: f(1) = 0

    Args:
        x: Аргумент функции

    Returns:
        Значение функции
    """
    w = 1 + (x - 1) / 4
    return np.sin(np.pi * w)**2 + (w - 1)**2 * (1 + 10 * np.sin(np.pi * w + 1)**2)


def levy_range() -> Tuple[float, float]:
    """Рекомендуемый диапазон для функции Леви"""
    return -10.0, 10.0


# Словарь доступных функций для удобного обращения
AVAILABLE_FUNCTIONS = {
    'rastrigin': (rastrigin_1d, rastrigin_range, 'Функция Растригина'),
    'ackley': (ackley_1d, ackley_range, 'Функция Экли'),
    'griewank': (griewank_1d, griewank_range, 'Функция Гриванка'),
    'schwefel': (schwefel_1d, schwefel_range, 'Функция Швефеля'),
    'multimodal1': (multimodal_1, multimodal_1_range, 'Многоэкстремальная функция 1'),
    'multimodal2': (multimodal_2, multimodal_2_range, 'Многоэкстремальная функция 2'),
    'shubert': (shubert_1d, shubert_range, 'Функция Шуберта'),
    'levy': (levy_1d, levy_range, 'Функция Леви'),
}


def get_function(name: str) -> Tuple:
    """
    Получить функцию и её рекомендуемый диапазон по имени

    Args:
        name: Название функции

    Returns:
        Кортеж (функция, диапазон, описание)
    """
    if name in AVAILABLE_FUNCTIONS:
        return AVAILABLE_FUNCTIONS[name]
    else:
        raise ValueError(f"Функция '{name}' не найдена. "
                         f"Доступные функции: {list(AVAILABLE_FUNCTIONS.keys())}")


def list_available_functions():
    """Вывести список доступных функций"""
    print("\nДоступные тестовые функции:")
    print("=" * 60)
    for key, (_, range_func, description) in AVAILABLE_FUNCTIONS.items():
        a, b = range_func()
        print(f"{key:15s} - {description:30s} [{a:.2f}, {b:.2f}]")
    print("=" * 60)
