import re


def prettify(string):
    new_string = re.sub('_', ' ', string)
    return new_string
