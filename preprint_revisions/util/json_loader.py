import json


def load_json(filename):
    """
    This function takes a json file and returns a dictionary
    Args:
        filename (str): name of json file to load data from
    Returns:
        data (dict): dictionary
    """

    with open(filename, "r") as f:
        data = json.load(f)
    return data


def save_to_json(data, filename):
    """
    This function takes a dictionary and saves it to a json file
    Args:
        data (list): dictionary
        filename (str): name of json file to save data to
    """

    with open(filename, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    pass
