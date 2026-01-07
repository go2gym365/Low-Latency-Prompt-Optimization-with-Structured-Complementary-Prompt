import json
import pickle

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_pickles(embeddings, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

def load_pickles(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def save_data(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

