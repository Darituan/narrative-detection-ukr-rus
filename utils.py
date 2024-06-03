import json
import re


def find_first_digit(input_string):
    match = re.search(r'\d', input_string)
    if match:
        return input_string[match.start()]
    else:
        return None


def get_json_substrings(text):
    return re.findall(r'\{.*?\}', text)
