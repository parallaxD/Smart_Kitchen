# -*- coding: utf-8 -*-
import sqlite3
import uuid
import sys

# Установка UTF-8 для вывода
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

def add_inventory():
    """Добавление или обновление инвентаря в базе."""
    try:
        conn = sqlite3.connect("cooking_system.db")
        conn.text_factory = str
        cursor = conn.cursor()
        inventory_data = [
            ("R1", "мука", "100", "г", "нескоропортящийся"),
            ("R1", "яйца", "1", "", "скоропортящийся"),
            ("R2", "мука", "50", "г", "нескоропортящийся"),
            ("R2", "молоко", "100", "мл", "скоропортящийся")
        ]
        for refrigerator_id, name, quantity, unit, type_ in inventory_data:
            cursor.execute(
                "SELECT inventory_id, quantity FROM Inventory WHERE refrigerator_id = ? AND ingredient_name = ? AND unit = ?",
                (refrigerator_id, name, unit)
            )
            result = cursor.fetchone()
            if result:
                new_quantity = float(result[1]) + float(quantity)
                cursor.execute(
                    "UPDATE Inventory SET quantity = ? WHERE inventory_id = ?",
                    (new_quantity, result[0])
                )
                print(f"Обновлен ингредиент '{name}' в холодильнике {refrigerator_id}: {new_quantity} {unit}")
            else:
                cursor.execute(
                    "INSERT INTO Inventory (inventory_id, refrigerator_id, ingredient_name, quantity, unit, type) VALUES (?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), refrigerator_id, name, quantity, unit, type_)
                )
                print(f"Добавлен ингредиент '{name}' в холодильник {refrigerator_id}")
        conn.commit()
        conn.close()
        print("Инвентарь обновлен")
    except sqlite3.Error as e:
        print(f"Ошибка базы данных: {e}")

if __name__ == "__main__":
    add_inventory()