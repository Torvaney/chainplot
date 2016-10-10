import re


def prettify(label):
    label = str(label)
    new_string = re.sub('_', ' ', label)
    new_string = new_string.capitalize()
    return new_string
