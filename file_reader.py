import json

def read_json(file_path):
    try:
        with open(file_path) as f:
            data = json.load(f)
        return data

    except:
        # print("File not found")
        pass
    return False