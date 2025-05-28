# -*- coding: utf-8 -*-
import sqlite3
import uuid
import sys

# Установка UTF-8 для вывода
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

def add_cookware():
    """Добавление или обновление посуды в базе."""
    try:
        conn = sqlite3.connect("cooking_system.db")
        conn.text_factory = str
        cursor = conn.cursor()
        cookware_data = [
            ("миска", 1),
            ("сковорода", 1),
            ("кастрюля", 0)
        ]
        for name, available in cookware_data:
            cursor.execute("SELECT cookware_id, available FROM Cookware WHERE name = ?", (name,))
            result = cursor.fetchone()
            if result:
                new_available = result[1] + available
                cursor.execute("UPDATE Cookware SET available = ? WHERE cookware_id = ?", (new_available, result[0]))
                print(f"Обновлена посуда '{name}': available = {new_available}")
            else:
                cookware_id = str(uuid.uuid4())
                cursor.execute("INSERT INTO Cookware (cookware_id, name, available) VALUES (?, ?, ?)", (cookware_id, name, available))
                print(f"Добавлена посуда '{name}' с available = {available}")
        conn.commit()
        conn.close()
        print("Посуда обновлена")
    except sqlite3.Error as e:
        print(f"Ошибка базы данных: {e}")

if __name__ == "__main__":
    add_cookware()