# -*- coding: utf-8 -*-
from flask import Flask, request, render_template
from main import RecipeProcessor
import sys
from contextlib import closing

# Установка UTF-8 для вывода
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)

# Инициализация процессора рецептов
recipe_processor = RecipeProcessor()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            recipe_text = request.form["recipe_text"]
            recipe_name = request.form["recipe_name"]
            num_portions = int(request.form["num_portions"])
            refrigerators = [fridge.strip() for fridge in request.form["refrigerators"].split(",")]

            # Обработка рецепта
            result = recipe_processor.process_recipe_text(recipe_text, recipe_name)

            # Планирование блюда
            meal_plan = recipe_processor.plan_meal(num_portions, refrigerators)

            return render_template("result.html",
                                   result=result,
                                   meal_plan=meal_plan)

        except Exception as e:
            return f"Произошла ошибка: {str(e)}", 500

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5001)