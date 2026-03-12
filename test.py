import json
import datasets
import os

def load_gsm8k_data():
    """
    Loads the GSM8K dataset (train split) and formats it into
    a list of dictionaries with 'input' and 'output' keys.
    """
    print("Loading GSM8K dataset from Hugging Face...")
    # Load the 'main' configuration of gsm8k, train split
    ds = datasets.load_dataset("gsm8k", "main", split="train")
    
    data_list = []
    for sample in ds:
        item = {
            "input": sample['question'],
            "output": sample['answer']
        }
        data_list.append(item)
        
    return data_list

def generate_json_file(data_list, filename="gsm8k_results.json"):
    """
    Wraps the data list in a dictionary with a "results" key
    and saves it to a JSON file.
    """
    # Create the required structure containing the "results" attribute
    final_structure = {
        "results": data_list
    }
    
    print(f"Writing {len(data_list)} items to {filename}...")
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # indent=2 makes the file human-readable
            json.dump(final_structure, f, indent=2, ensure_ascii=False)
        print(f"Success! File saved as: {os.path.abspath(filename)}")
    except IOError as e:
        print(f"Error writing to file: {e}")

if __name__ == "__main__":
    # 1. Get the data using the provided logic
    formatted_data = load_gsm8k_data()
    
    # 2. Save the data to the JSON file
    generate_json_file(formatted_data, "gsm8k_data.json")