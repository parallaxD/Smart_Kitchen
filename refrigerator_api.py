# -*- coding: utf-8 -*-
from flask import Flask, jsonify
import sqlite3
import sys

# Установка UTF-8 для вывода
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)

@app.route("/inventory/<refrigerator_id>")
def get_inventory(refrigerator_id):
    try:
        conn = sqlite3.connect("cooking_system.db")
        conn.text_factory = str
        cursor = conn.cursor()
        cursor.execute("SELECT ingredient_name, quantity, unit, type FROM Inventory WHERE refrigerator_id = ?", (refrigerator_id,))
        inventory = [{"name": row[0], "quantity": str(row[1]), "unit": row[2], "type": row[3]} for row in cursor.fetchall()]
        conn.close()
        return jsonify(inventory)
    except sqlite3.Error as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000)