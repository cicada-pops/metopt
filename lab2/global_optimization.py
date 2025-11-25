"""
Модуль для поиска глобального экстремума липшицевой функции
методом Пиявского (метод ломаных)
"""

import numpy as np
import time
from typing import Callable, Tuple, List, Dict


class LipschitzOptimizer:
    """
    Класс для поиска глобального минимума липшицевой функции
    на заданном отрезке методом Пиявского
    """

    def __init__(self, func: Callable[[float], float], a: float, b: float, eps: float = 0.01):
        """
        Инициализация оптимизатора

        Args:
            func: Целевая функция для минимизации
            a: Левая граница отрезка
            b: Правая граница отрезка
            eps: Требуемая точность
        """
        self.func = func
        self.a = a
        self.b = b
        self.eps = eps

        # История вычислений
        self.points = []  # Пробные точки
        self.values = []  # Значения функции в точках
        self.iterations = 0
        self.time_elapsed = 0

        # Результат
        self.x_min = None
        self.f_min = None

    def estimate_lipschitz_constant(self) -> float:
        """
        Оценка константы Липшица по имеющимся точкам

        Returns:
            Оценка константы Липшица
        """
        if len(self.points) < 2:
            return 1.0

        max_slope = 0.0
        for i in range(len(self.points)):
            for j in range(i + 1, len(self.points)):
                x1, x2 = self.points[i], self.points[j]
                f1, f2 = self.values[i], self.values[j]
                slope = abs(f2 - f1) / abs(x2 - x1)
                max_slope = max(max_slope, slope)

        # Увеличиваем оценку для надежности
        return max_slope * 1.5 if max_slope > 0 else 1.0

    def compute_characteristic(self, i: int, L: float) -> float:
        """
        Вычисление характеристики интервала (i-1, i)

        Args:
            i: Индекс правой точки интервала
            L: Константа Липшица

        Returns:
            Значение характеристики
        """
        x_prev, x_curr = self.points[i-1], self.points[i]
        f_prev, f_curr = self.values[i-1], self.values[i]

        # Характеристика по методу Пиявского
        R = L * (x_curr - x_prev) + (f_curr - f_prev)**2 / (L * (x_curr - x_prev)) - 2 * (f_curr + f_prev)

        return R

    def select_next_point(self, L: float) -> Tuple[int, float]:
        """
        Выбор следующей точки для испытания

        Args:
            L: Константа Липшица

        Returns:
            Индекс интервала и новая точка
        """
        max_R = -np.inf
        max_interval = 1

        # Находим интервал с максимальной характеристикой
        for i in range(1, len(self.points)):
            R = self.compute_characteristic(i, L)
            if R > max_R:
                max_R = R
                max_interval = i

        # Вычисляем новую точку в интервале с максимальной характеристикой
        i = max_interval
        x_prev, x_curr = self.points[i-1], self.points[i]
        f_prev, f_curr = self.values[i-1], self.values[i]

        x_new = (x_prev + x_curr) / 2 - (f_curr - f_prev) / (2 * L)

        # Убеждаемся, что точка внутри интервала
        x_new = max(x_prev, min(x_curr, x_new))

        return i, x_new

    def optimize(self) -> Dict:
        """
        Выполнение оптимизации

        Returns:
            Словарь с результатами оптимизации
        """
        start_time = time.time()

        # Инициализация: вычисляем функцию на концах отрезка
        self.points = [self.a, self.b]
        self.values = [self.func(self.a), self.func(self.b)]
        self.iterations = 2

        # Основной цикл
        while True:
            # Сортируем точки по возрастанию x
            sorted_indices = np.argsort(self.points)
            self.points = [self.points[i] for i in sorted_indices]
            self.values = [self.values[i] for i in sorted_indices]

            # Оценка константы Липшица
            L = self.estimate_lipschitz_constant()

            # Выбор следующей точки
            interval_idx, x_new = self.select_next_point(L)

            # Проверка критерия остановки
            x_prev = self.points[interval_idx - 1]
            x_curr = self.points[interval_idx]

            if abs(x_curr - x_prev) < self.eps:
                break

            # Вычисление функции в новой точке
            f_new = self.func(x_new)

            # Добавление новой точки
            self.points.insert(interval_idx, x_new)
            self.values.insert(interval_idx, f_new)
            self.iterations += 1

            # Защита от бесконечного цикла
            if self.iterations > 10000:
                break

        # Конец оптимизации
        self.time_elapsed = time.time() - start_time

        # Находим минимум
        min_idx = np.argmin(self.values)
        self.x_min = self.points[min_idx]
        self.f_min = self.values[min_idx]

        return {
            'x_min': self.x_min,
            'f_min': self.f_min,
            'iterations': self.iterations,
            'time': self.time_elapsed,
            'points': self.points.copy(),
            'values': self.values.copy(),
            'lipschitz_constant': self.estimate_lipschitz_constant()
        }

    def get_piecewise_function(self, num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Построение ломаной функции (оценочной функции) методом Пиявского

        Args:
            num_points: Количество точек для построения

        Returns:
            Массивы x и y для построения ломаной
        """
        if len(self.points) < 2:
            return np.array([]), np.array([])

        L = self.estimate_lipschitz_constant()
        
        # Сортируем точки
        sorted_indices = np.argsort(self.points)
        sorted_points = [self.points[i] for i in sorted_indices]
        sorted_values = [self.values[i] for i in sorted_indices]
        
        x_all = []
        y_all = []
        
        # Строим ломаную для каждого интервала между соседними точками
        for i in range(len(sorted_points) - 1):
            x_left, x_right = sorted_points[i], sorted_points[i + 1]
            f_left, f_right = sorted_values[i], sorted_values[i + 1]
            
            # Находим точку минимума V-образной ломаной на этом интервале
            x_star = (x_left + x_right) / 2 - (f_right - f_left) / (2 * L)
            x_star = max(x_left, min(x_right, x_star))
            
            # Значение в точке минимума
            y_star = f_left - L * (x_star - x_left)
            
            # Добавляем точки ломаной для этого интервала
            n_points_interval = max(2, int(num_points * (x_right - x_left) / (self.b - self.a)))
            
            # Левая часть (от x_left до x_star)
            if not np.isclose(x_left, x_star):
                x_left_part = np.linspace(x_left, x_star, n_points_interval // 2, endpoint=False)
                y_left_part = f_left - L * (x_left_part - x_left)
                x_all.extend(x_left_part)
                y_all.extend(y_left_part)
            
            # Правая часть (от x_star до x_right)
            if not np.isclose(x_star, x_right):
                x_right_part = np.linspace(x_star, x_right, n_points_interval // 2 + 1)
                y_right_part = f_right - L * (x_right - x_right_part)
                x_all.extend(x_right_part)
                y_all.extend(y_right_part)
        
        # Добавляем последнюю точку
        if len(sorted_points) > 0:
            if len(x_all) == 0 or not np.isclose(x_all[-1], sorted_points[-1]):
                x_all.append(sorted_points[-1])
                y_all.append(sorted_values[-1])

        return np.array(x_all), np.array(y_all)
