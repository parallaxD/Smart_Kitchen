# -*- coding: utf-8 -*-
import sqlite3
import spacy
from transformers import pipeline
import re
import uuid
from typing import List, Dict
import logging
import os

# Установка UTF-8 для консоли в Windows
if os.name == 'nt':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

# Отключение предупреждения о символических ссылках
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Настройка логирования с UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Проверка зависимостей
try:
    import torch
    logger.info("PyTorch установлен: версия %s", torch.__version__)
except ImportError:
    logger.error("PyTorch не установлен. Установите с помощью: pip install torch")
    exit(1)

try:
    nlp = spacy.load("ru_core_news_sm")
    logger.info("Модель SpaCy 'ru_core_news_sm' успешно загружена")
except OSError as e:
    logger.error("Не удалось загрузить модель SpaCy 'ru_core_news_sm'. Установите с помощью: python -m spacy download ru_core_news_sm")
    logger.error(str(e))
    exit(1)

try:
    intent_classifier = pipeline("text-classification", model="DeepPavlov/rubert-base-cased")
    logger.info("Классификатор намерений Transformers успешно загружен")
except Exception as e:
    logger.error("Не удалось загрузить модель Transformers. Убедитесь, что PyTorch или TensorFlow установлены.")
    logger.error(str(e))
    exit(1)

# Инициализация базы данных
def init_database():
    """Инициализация базы данных SQLite с поддержкой UTF-8."""
    try:
        conn = sqlite3.connect("cooking_system.db")
        conn.text_factory = str  # Убедимся, что текст возвращается как UTF-8
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Recipes (
                recipe_id TEXT PRIMARY KEY,
                name TEXT,
                raw_text TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Ingredients (
                ingredient_id TEXT PRIMARY KEY,
                recipe_id TEXT,
                name TEXT,
                quantity TEXT,
                unit TEXT,
                type TEXT,
                FOREIGN KEY (recipe_id) REFERENCES Recipes(recipe_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Steps (
                step_id TEXT PRIMARY KEY,
                recipe_id TEXT,
                step_number INTEGER,
                action TEXT,
                equipment TEXT,
                duration TEXT,
                FOREIGN KEY (recipe_id) REFERENCES Recipes(recipe_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Rules (
                rule_id TEXT PRIMARY KEY,
                recipe_id TEXT,
                rule_text TEXT,
                FOREIGN KEY (recipe_id) REFERENCES Recipes(recipe_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Inventory (
                inventory_id TEXT PRIMARY KEY,
                refrigerator_id TEXT,
                ingredient_name TEXT,
                quantity REAL,
                unit TEXT,
                type TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Cookware (
                cookware_id TEXT PRIMARY KEY,
                name TEXT UNIQUE,
                available INTEGER
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Orders (
                order_id TEXT PRIMARY KEY,
                ingredient_name TEXT,
                quantity TEXT,
                unit TEXT,
                status TEXT
            )
        """)

        conn.commit()
        logger.info("База данных успешно инициализирована")
    except sqlite3.Error as e:
        logger.error("Ошибка инициализации базы данных: %s", str(e))
        exit(1)
    finally:
        conn.close()

def check_recipe_exists(recipe_name: str) -> str:
    """Проверка, существует ли рецепт с заданным названием."""
    try:
        conn = sqlite3.connect("cooking_system.db")
        conn.text_factory = str
        cursor = conn.cursor()
        cursor.execute("SELECT recipe_id FROM Recipes WHERE name = ?", (recipe_name,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    except sqlite3.Error as e:
        logger.error("Ошибка проверки рецепта: %s", str(e))
        return None

def retrieve_recipe_data(recipe_id: str) -> Dict:
    """Извлечение данных рецепта из базы."""
    try:
        conn = sqlite3.connect("cooking_system.db")
        conn.text_factory = str
        cursor = conn.cursor()

        cursor.execute("SELECT name, raw_text FROM Recipes WHERE recipe_id = ?", (recipe_id,))
        recipe_data = cursor.fetchone()
        if not recipe_data:
            return {}

        cursor.execute("SELECT name, quantity, unit, type FROM Ingredients WHERE recipe_id = ?", (recipe_id,))
        ingredients = [{"name": row[0], "quantity": row[1], "unit": row[2], "type": row[3]} for row in cursor.fetchall()]

        cursor.execute("SELECT step_number, action, equipment, duration FROM Steps WHERE recipe_id = ? ORDER BY step_number", (recipe_id,))
        steps = [{"step_number": row[0], "action": row[1], "equipment": row[2], "duration": row[3]} for row in cursor.fetchall()]
        actions = [step["action"] for step in steps]
        equipment = [step["equipment"] for step in steps if step["equipment"]]

        cursor.execute("SELECT rule_text FROM Rules WHERE recipe_id = ?", (recipe_id,))
        rules = [row[0] for row in cursor.fetchall()]

        conn.close()
        return {
            "recipe_id": recipe_id,
            "name": recipe_data[0],  # Добавляем имя рецепта
            "entities": {
                "ingredients": ingredients,
                "actions": actions,
                "equipment": equipment,
                "steps": steps
            },
            "intents": ["приготовить_блюдо"],
            "rules": rules,
            "raw_text": recipe_data[1]
        }
    except sqlite3.Error as e:
        logger.error("Ошибка извлечения данных рецепта: %s", str(e))
        return {}

def update_cookware(cookware_name: str, quantity: int = 1):
    """Обновление количества доступной посуды или добавление новой."""
    try:
        conn = sqlite3.connect("cooking_system.db")
        conn.text_factory = str
        cursor = conn.cursor()
        cursor.execute("SELECT cookware_id, available FROM Cookware WHERE name = ?", (cookware_name,))
        result = cursor.fetchone()
        if result:
            new_available = result[1] + quantity
            cursor.execute("UPDATE Cookware SET available = ? WHERE cookware_id = ?", (new_available, result[0]))
            logger.info("Обновлено количество посуды '%s': available = %d", cookware_name, new_available)
        else:
            cookware_id = str(uuid.uuid4())
            cursor.execute("INSERT INTO Cookware (cookware_id, name, available) VALUES (?, ?, ?)", (cookware_id, cookware_name, quantity))
            logger.info("Добавлена новая посуда '%s' с available = %d", cookware_name, quantity)
        conn.commit()
    except sqlite3.Error as e:
        logger.error("Ошибка обновления посуды: %s", str(e))
    finally:
        conn.close()

def filter_non_essential(text: str) -> str:
    """Удаление несущественной информации из текста рецепта."""
    narrative_patterns = [
        r"Чтобы приготовить вкусный.*?,",
        r"Этот рецепт идеально подходит для.*?,",
        r"Наслаждайтесь этим.*?\."
    ]
    cleaned_text = text
    for pattern in narrative_patterns:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)
    return cleaned_text.strip()

def extract_entities(text: str) -> Dict:
    """Извлечение ингредиентов, количеств, действий и посуды с помощью SpaCy."""
    doc = nlp(text)
    ingredients = []
    actions = []
    equipment = []
    step_number = 0
    steps = []

    ingredient_keywords = ["мука", "яйца", "молоко", "соль", "масло", "сироп", "сахар", "вода"]
    action_keywords = ["смешать", "взбить", "соединить", "разогреть", "жарить", "вылить", "подавать", "варить"]
    equipment_keywords = ["миска", "сковорода", "кастрюля", "ложка", "венчик"]

    sentences = [sent.text for sent in doc.sents]

    for sent in sentences:
        sent_doc = nlp(sent)
        for token in sent_doc:
            if token.text.lower() in ingredient_keywords:
                quantity = ""
                unit = ""
                for child in token.head.children:
                    if child.text.lower() in ["г", "мл", "щепотка", "л", "шт"] or child.ent_type_ == "QUANTITY":
                        if child.text.isdigit() or re.match(r"\d+\.?\d*", child.text):
                            quantity = child.text
                        else:
                            unit = child.text
                ingredients.append({"name": token.text, "quantity": quantity, "unit": unit})

            if token.pos_ == "VERB" and token.text.lower() in action_keywords:
                actions.append(token.text)
                equipment_in_sent = [t.text for t in sent_doc if t.text.lower() in equipment_keywords]
                duration = re.search(r"\d+\s*(минуты?|секунды?)", sent)
                steps.append({
                    "step_number": step_number + 1,
                    "action": token.text,
                    "equipment": equipment_in_sent[0] if equipment_in_sent else None,
                    "duration": duration.group(0) if duration else None
                })
                step_number += 1

            if token.text.lower() in equipment_keywords:
                equipment.append(token.text)

    return {
        "ingredients": ingredients,
        "actions": actions,
        "equipment": equipment,
        "steps": steps
    }

def detect_intents(text: str) -> List[str]:
    """Определение намерений с помощью модели Transformers."""
    try:
        result = intent_classifier(text)
        intents = ["приготовить_блюдо"] if result[0]["label"] == "POSITIVE" else []
        return intents
    except Exception as e:
        logger.error("Ошибка определения намерений: %s", str(e))
        return []

def extract_causal_rules(entities: Dict, text: str) -> List[str]:
    """Извлечение правил вида 'Если...То'."""
    rules = []
    ingredients = entities["ingredients"]
    if ingredients:
        ingredient_names = [ing["name"] for ing in ingredients]
        rules.append(f"Если есть {', '.join(ingredient_names)}, то можно приготовить блюдо.")
    return rules

def resolve_contradictions(entities: Dict, text: str) -> Dict:
    """Разрешение противоречий в количестве или единицах измерения."""
    ingredient_names = [ing["name"] for ing in entities["ingredients"]]
    if len(ingredient_names) != len(set(ingredient_names)):
        unique_ingredients = []
        seen = set()
        for ing in entities["ingredients"]:
            if ing["name"] not in seen:
                unique_ingredients.append(ing)
                seen.add(ing["name"])
        entities["ingredients"] = unique_ingredients
    return entities

def store_in_database(result: Dict, recipe_name: str):
    """Сохранение данных рецепта в базе SQLite."""
    try:
        conn = sqlite3.connect("cooking_system.db")
        conn.text_factory = str
        cursor = conn.cursor()

        recipe_id = result["recipe_id"]
        cursor.execute(
            "INSERT INTO Recipes (recipe_id, name, raw_text) VALUES (?, ?, ?)",
            (recipe_id, recipe_name, result["raw_text"])
        )

        for ing in result["entities"]["ingredients"]:
            ingredient_id = str(uuid.uuid4())
            ing_type = "скоропортящийся" if ing["name"].lower() in ["яйца", "молоко", "масло"] else "нескоропортящийся"
            cursor.execute(
                "INSERT INTO Ingredients (ingredient_id, recipe_id, name, quantity, unit, type) VALUES (?, ?, ?, ?, ?, ?)",
                (ingredient_id, recipe_id, ing["name"], ing["quantity"], ing["unit"], ing_type)
            )

        for step in result["entities"]["steps"]:
            step_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO Steps (step_id, recipe_id, step_number, action, equipment, duration) VALUES (?, ?, ?, ?, ?, ?)",
                (step_id, recipe_id, step["step_number"], step["action"], step["equipment"], step["duration"])
            )

        for rule in result["rules"]:
            rule_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO Rules (rule_id, recipe_id, rule_text) VALUES (?, ?, ?)",
                (rule_id, recipe_id, rule)
            )

        conn.commit()
        logger.info("Данные рецепта успешно сохранены для: %s", recipe_name)
    except sqlite3.Error as e:
        logger.error("Ошибка сохранения данных в базе: %s", str(e))
    finally:
        conn.close()

def process_recipe_text(text: str, recipe_name: str) -> Dict:
    """Обработка текста рецепта или извлечение существующего рецепта."""
    init_database()

    existing_recipe_id = check_recipe_exists(recipe_name)
    if existing_recipe_id:
        logger.info("Рецепт '%s' уже существует с ID: %s", recipe_name, existing_recipe_id)
        return retrieve_recipe_data(existing_recipe_id)

    cleaned_text = filter_non_essential(text)
    logger.info("Текст успешно очищен")
    entities = extract_entities(cleaned_text)
    intents = detect_intents(cleaned_text)
    rules = extract_causal_rules(entities, cleaned_text)
    entities = resolve_contradictions(entities, cleaned_text)
    result = {
        "recipe_id": str(uuid.uuid4()),
        "name": recipe_name,
        "entities": entities,
        "intents": intents,
        "rules": rules,
        "raw_text": cleaned_text
    }
    store_in_database(result, recipe_name)
    return result

def plan_meal(num_portions: int, refrigerator_ids: List[str]) -> Dict:
    """Планирование блюда на основе инвентаря и количества порций."""
    conn = sqlite3.connect("cooking_system.db")
    conn.text_factory = str
    cursor = conn.cursor()

    cursor.execute(
        "SELECT ingredient_name, SUM(quantity) as total_quantity, unit, type "
        "FROM Inventory WHERE refrigerator_id IN ({}) GROUP BY ingredient_name, unit".format(
            ','.join('?' for _ in refrigerator_ids)
        ),
        refrigerator_ids
    )
    inventory = {row[0]: {"quantity": row[1], "unit": row[2], "type": row[3]} for row in cursor.fetchall()}

    cursor.execute("SELECT name FROM Cookware WHERE available > 0")
    available_cookware = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT recipe_id, name FROM Recipes")
    recipes = cursor.fetchall()
    suitable_recipes = []
    missing_ingredients = []

    for recipe_id, recipe_name in recipes:
        cursor.execute("SELECT name, quantity, unit FROM Ingredients WHERE recipe_id = ?", (recipe_id,))
        required_ingredients = cursor.fetchall()
        cursor.execute("SELECT equipment FROM Steps WHERE recipe_id = ?", (recipe_id,))
        required_equipment = [row[0] for row in cursor.fetchall() if row[0]]

        can_prepare = True
        recipe_missing = []
        for ing_name, ing_quantity, ing_unit in required_ingredients:
            ing_quantity = float(ing_quantity) * num_portions / 4 if ing_quantity else 0
            if ing_name in inventory and inventory[ing_name]["unit"] == ing_unit:
                if inventory[ing_name]["quantity"] < ing_quantity:
                    recipe_missing.append({
                        "name": ing_name,
                        "quantity": ing_quantity - inventory[ing_name]["quantity"],
                        "unit": ing_unit
                    })
                    can_prepare = False
            else:
                recipe_missing.append({"name": ing_name, "quantity": ing_quantity, "unit": ing_unit})
                can_prepare = False

        if not all(equip in available_cookware for equip in required_equipment):
            can_prepare = False

        if can_prepare:
            suitable_recipes.append({"recipe_id": recipe_id, "name": recipe_name})
        else:
            missing_ingredients.append({"recipe_id": recipe_id, "name": recipe_name, "missing": recipe_missing})

    conn.close()
    return {"suitable_recipes": suitable_recipes, "missing_ingredients": missing_ingredients}

def generate_orders(missing_ingredients: List[Dict]):
    """Создание заказов для недостающих ингредиентов."""
    conn = sqlite3.connect("cooking_system.db")
    conn.text_factory = str
    cursor = conn.cursor()
    for recipe in missing_ingredients:
        for ing in recipe["missing"]:
            order_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO Orders (order_id, ingredient_name, quantity, unit, status) VALUES (?, ?, ?, ?, ?)",
                (order_id, ing["name"], str(ing["quantity"]), ing["unit"], "в ожидании")
            )
    conn.commit()
    conn.close()
    logger.info("Заказы сформированы")

def update_inventory(recipe_id: str, num_portions: int):
    """Обновление инвентаря после приготовления."""
    try:
        conn = sqlite3.connect("cooking_system.db")
        conn.text_factory = str
        cursor = conn.cursor()
        cursor.execute("SELECT name, quantity, unit FROM Ingredients WHERE recipe_id = ?", (recipe_id,))
        ingredients = cursor.fetchall()
        for ing_name, ing_quantity, ing_unit in ingredients:
            ing_quantity = float(ing_quantity) * num_portions / 4 if ing_quantity else 0
            cursor.execute(
                "UPDATE Inventory SET quantity = quantity - ? WHERE ingredient_name = ? AND unit = ?",
                (ing_quantity, ing_name, ing_unit)
            )
        conn.commit()
        logger.info("Инвентарь обновлен для рецепта %s", recipe_id)
    except sqlite3.Error as e:
        logger.error("Ошибка обновления инвентаря: %s", str(e))
    finally:
        conn.close()

def sync_inventory(refrigerator_id: str, api_url: str = "http://localhost:5000/inventory"):
    """Синхронизация инвентаря с API холодильника."""
    try:
        import requests
        response = requests.get(f"{api_url}/{refrigerator_id}")
        inventory_data = response.json()
        conn = sqlite3.connect("cooking_system.db")
        conn.text_factory = str
        cursor = conn.cursor()
        for item in inventory_data:
            cursor.execute(
                "SELECT inventory_id, quantity FROM Inventory WHERE refrigerator_id = ? AND ingredient_name = ? AND unit = ?",
                (refrigerator_id, item["name"], item["unit"])
            )
            result = cursor.fetchone()
            if result:
                new_quantity = float(result[1]) + float(item["quantity"])
                cursor.execute(
                    "UPDATE Inventory SET quantity = ? WHERE inventory_id = ?",
                    (new_quantity, result[0])
                )
                logger.info("Обновлен ингредиент '%s' в холодильнике %s: quantity = %s %s", item["name"], refrigerator_id, new_quantity, item["unit"])
            else:
                cursor.execute(
                    "INSERT INTO Inventory (inventory_id, refrigerator_id, ingredient_name, quantity, unit, type) VALUES (?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), refrigerator_id, item["name"], item["quantity"], item["unit"], item["type"])
                )
                logger.info("Добавлен ингредиент '%s' в холодильник %s", item["name"], refrigerator_id)
        conn.commit()
        conn.close()
        logger.info("Инвентарь для холодильника %s синхронизирован", refrigerator_id)
    except requests.RequestException as e:
        logger.error("Ошибка синхронизации инвентаря: %s", str(e))

if __name__ == "__main__":
    sample_text = """
    Чтобы приготовить блины на 4 человека, вам понадобится 200 г муки, 2 яйца, 
    300 мл молока и щепотка соли. Сначала смешайте муку и соль в миске. Затем 
    взбейте яйца с молоком и соедините с сухими ингредиентами. Разогрейте сковороду 
    на среднем огне, добавьте немного сливочного масла и вылейте тесто для блинов. 
    Жарьте каждую сторону 2 минуты. Подавайте с сиропом.
    """
    result = process_recipe_text(sample_text, "Блины")
    print("Обработанный рецепт:")
    print(f"Название: {result['name']}")
    print(f"ID рецепта: {result['recipe_id']}")
    print(f"Сущности: {result['entities']}")
    print(f"Намерения: {result['intents']}")
    print(f"Правила: {result['rules']}")

    meal_plan = plan_meal(num_portions=4, refrigerator_ids=["R1", "R2"])
    print("План блюд:", meal_plan)
    generate_orders(meal_plan["missing_ingredients"])
    if meal_plan["suitable_recipes"]:
        update_inventory(meal_plan["suitable_recipes"][0]["recipe_id"], num_portions=4)