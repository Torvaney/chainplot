import re


def prettify(string):
    new_string = re.sub('_', ' ', string)
    new_string = new_string.capitalize()
    return new_string
