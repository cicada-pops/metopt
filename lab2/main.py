"""
Главная программа для поиска глобального экстремума липшицевой функции
Использует метод Пиявского (метод ломаных)
"""

import numpy as np
from global_optimization import LipschitzOptimizer
from visualization import plot_optimization_results, print_results
from test_functions import get_function, list_available_functions, AVAILABLE_FUNCTIONS


def parse_function_string(func_str: str):
    """
    Преобразование строки функции в вычислимую функцию

    Args:
        func_str: Строка с математическим выражением (например, "x + sin(3.14159*x)")

    Returns:
        Функция, принимающая один аргумент
    """
    # Создаем безопасное окружение для eval
    safe_dict = {
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'pi': np.pi,
        'e': np.e,
    }

    def func(x):
        safe_dict['x'] = x
        try:
            return eval(func_str, {"__builtins__": {}}, safe_dict)
        except Exception as e:
            raise ValueError(f"Ошибка при вычислении функции: {e}")

    return func


def run_optimization_from_string(func_str: str, a: float, b: float, eps: float = 0.01):
    """
    Запуск оптимизации для функции, заданной строкой

    Args:
        func_str: Строка с математическим выражением
        a: Левая граница отрезка
        b: Правая граница отрезка
        eps: Требуемая точность
    """
    print(f"\n{'='*70}")
    print(f"Оптимизация функции: f(x) = {func_str}")
    print(f"Отрезок: [{a}, {b}]")
    print(f"Точность: {eps}")
    print(f"{'='*70}\n")

    # Преобразование строки в функцию
    func = parse_function_string(func_str)

    # Создание оптимизатора и запуск
    optimizer = LipschitzOptimizer(func, a, b, eps)
    result = optimizer.optimize()

    # Вывод результатов
    print_results(result)

    # Визуализация
    plot_optimization_results(func, result, a, b, f"f(x) = {func_str}")


def run_optimization_from_library(func_name: str, eps: float = 0.01,
                                   custom_range: tuple = None):
    """
    Запуск оптимизации для функции из библиотеки тестовых функций

    Args:
        func_name: Название функции из библиотеки
        eps: Требуемая точность
        custom_range: Пользовательский диапазон (опционально)
    """
    # Получение функции и её диапазона
    func, range_func, description = get_function(func_name)

    if custom_range is None:
        a, b = range_func()
    else:
        a, b = custom_range

    print(f"\n{'='*70}")
    print(f"Оптимизация: {description}")
    print(f"Отрезок: [{a}, {b}]")
    print(f"Точность: {eps}")
    print(f"{'='*70}\n")

    # Создание оптимизатора и запуск
    optimizer = LipschitzOptimizer(func, a, b, eps)
    result = optimizer.optimize()

    # Вывод результатов
    print_results(result)

    # Визуализация
    plot_optimization_results(func, result, a, b, description)


def demo_all_functions():
    """
    Демонстрация работы на всех доступных тестовых функциях
    """
    print("\n" + "="*70)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ НА ВСЕХ ТЕСТОВЫХ ФУНКЦИЯХ")
    print("="*70)

    eps = 0.01

    # Список функций для демонстрации (выбираем наиболее интересные)
    demo_functions = ['rastrigin', 'ackley', 'multimodal1', 'shubert']

    for func_name in demo_functions:
        run_optimization_from_library(func_name, eps)
        print("\n" + "-"*70 + "\n")


def interactive_mode():
    """
    Интерактивный режим для ввода пользовательских функций
    """
    print("\n" + "="*70)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("="*70)
    print("\nВведите параметры для оптимизации:")

    # Ввод функции
    func_str = input("Функция f(x) (например, 'x + sin(3.14159*x)'): ").strip()

    # Ввод границ отрезка
    try:
        a = float(input("Левая граница отрезка (a): ").strip())
        b = float(input("Правая граница отрезка (b): ").strip())
        eps = float(input("Точность (eps, например 0.01): ").strip() or "0.01")
    except ValueError:
        print("Ошибка: Неверный формат числа!")
        return

    # Запуск оптимизации
    try:
        run_optimization_from_string(func_str, a, b, eps)
    except Exception as e:
        print(f"Ошибка при выполнении оптимизации: {e}")


def main():
    """
    Главная функция программы
    """
    print("\n" + "="*70)
    print("ПРОГРАММА ПОИСКА ГЛОБАЛЬНОГО ЭКСТРЕМУМА ЛИПШИЦЕВОЙ ФУНКЦИИ")
    print("Метод Пиявского (метод ломаных)")
    print("="*70)

    while True:
        print("\nВыберите режим работы:")
        print("1 - Оптимизация функции из библиотеки")
        print("2 - Ввод собственной функции")
        print("3 - Демонстрация на всех тестовых функциях")
        print("4 - Список доступных функций")
        print("0 - Выход")

        choice = input("\nВаш выбор: ").strip()

        if choice == '1':
            list_available_functions()
            func_name = input("\nВведите название функции: ").strip()
            eps_input = input("Введите точность (по умолчанию 0.01): ").strip()
            eps = float(eps_input) if eps_input else 0.01

            try:
                run_optimization_from_library(func_name, eps)
            except ValueError as e:
                print(f"Ошибка: {e}")
            except Exception as e:
                print(f"Произошла ошибка: {e}")

        elif choice == '2':
            interactive_mode()

        elif choice == '3':
            demo_all_functions()

        elif choice == '4':
            list_available_functions()

        elif choice == '0':
            print("\nЗавершение работы программы. До свидания!")
            break

        else:
            print("Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()
