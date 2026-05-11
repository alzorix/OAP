
import subprocess
import sys
from pathlib import Path

MODULES = {
    "analyze":   "DataAnalyze/main.py",
    "preprocess":"DataPreprocessing/main.py",
    "fetcher":   "NewsFetcher/main.py",
}

def show_menu():
    print("\n=== Выбор модуля ===")
    print("1. DataAnalyze (анализ обработанных данных)")
    print("2. DataPreprocessing (AI обработка)")
    print("3. NewsFetcher (Парсинг новостей)")
    print("0. Выход")
    choice = input("Ваш выбор: ").strip()
    if choice == "1":
        return "analyze"
    elif choice == "2":
        return "preprocess"
    elif choice == "3":
        return "fetcher"
    elif choice == "0":
        exit()
    else:
        print("Неверный ввод.")
        return show_menu()

def run_module(module_name):
    if module_name not in MODULES:
        print(f"Ошибка: неизвестный модуль '{module_name}'")
        print(f"Доступные: {', '.join(MODULES.keys())}")
        return False

    script_path = Path(__file__).parent / MODULES[module_name]
    if not script_path.exists():
        print(f"Ошибка: файл {script_path} не найден")
        return False

    cmd = [sys.executable, str(script_path)]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении {module_name}: {e}")
        return False
    return True

def main():
    # Если переданы аргументы командной строки
    if len(sys.argv) > 1:
        module = sys.argv[1]
        run_module(module)
    else:
        module = show_menu()
        if module is None:
            print("Выход.")
            return
        run_module(module)

if __name__ == "__main__":
    main()