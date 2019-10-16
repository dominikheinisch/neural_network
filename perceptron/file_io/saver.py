import json


def save(file, data):
    with open(file, 'w') as f:
        json.dump(data, f)
