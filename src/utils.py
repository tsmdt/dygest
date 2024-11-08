import json
import re


def is_valid_json(json_string):
    try:
        json.loads(json_string)
        return True
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        return False

def fix_json(json_string):
    # 1. Remove trailing commas before closing brackets/braces
    json_string = re.sub(r',\s*([\]}])', r'\1', json_string)

    # 2. Add missing commas between JSON objects in arrays
    json_string = re.sub(r'}\s*{', '}, {', json_string)

    # 3. Add missing commas between key-value pairs within objects
    json_string = re.sub(r'(".*?":\s*".*?")\s*(".*?":)', r'\1, \2', json_string)

    # 4. Ensure that keys are enclosed in double quotes
    json_string = re.sub(r'(\w+):', r'"\1":', json_string)

    # 5. Replace single quotes with double quotes
    json_string = json_string.replace("'", '"')

    return json_string

def validate_and_fix_json(json_string):
    if is_valid_json(json_string):
        return json.loads(json_string)
    
    print("Attempting to fix JSON...")
    fixed_json_str = fix_json(json_string)
    
    if is_valid_json(fixed_json_str):
        print("JSON fixed successfully.\n")
        return json.loads(fixed_json_str)
    else:
        print("Failed to fix JSON.")
        print(json_string)
        return None

def validate_groq_summaries(llm_result):
    try:
        summaries = validate_and_fix_json(llm_result)
    except json.JSONDecodeError:
        print("Error decoding JSON from Groq result.\n")
        print(llm_result)
    return summaries

def print_entities(entities: list[dict]) -> None:
    [print(f"{item['entity']} → {item['category']}") for item in entities]

def print_summaries(summaries: list[dict]) -> None:
    for idx, item in enumerate(summaries):
        print(f"[{idx + 1}] {item['topic']}: {item['summary']} (LOCATION: {item['location']})\n")
