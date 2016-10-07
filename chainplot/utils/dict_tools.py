import re


def replace_dict(input_dict, val=None, replacement=''):
    """ Search and replace dict values """
    output_dict = input_dict.copy()
    for k, v in output_dict.items():
        if v is val:
            output_dict[k] = replacement
    return output_dict


def split_kwargs(kwargs_dict, prefix='shadow_'):
    """
    Splits dictionary into two new dicts by checking the keys for a given prefix. Those key-value pairs
     with the prefix will be added to a new dictionary with prefix removed from the keys.
    :param kwargs_dict: original dict to be checked and split. As dict
    :param prefix: prefix to search keys for. As string
    :return: copy of original dict with preficed keys removed, new dict based on prefixed keys
    """
    input_dict = kwargs_dict.copy()
    output_dict = {}
    shadow_dict = {}

    for k, v in input_dict.items():
        if bool(re.match(prefix + '.*', k)):  # find keys with prefix
            shadow_k = re.sub(prefix, '', k)  # remove prefix
            shadow_dict[shadow_k] = v
        else:
            output_dict[k] = v

    return output_dict, shadow_dict


def to_kwargs(**kwargs):
    return kwargs


def britishdict(argdict):
    """
    Creates a new duplicate dict in which keys with british spellings are convered to american spellings.
    """
    outdict = argdict.copy()

    for key, value in outdict.items():
        if bool(re.match('.*colour.*', key)):
            amerikey = re.sub('colour', 'color', key)
            outdict[amerikey] = value
            outdict.pop(key, 0)

    return outdict


def combine_dict(a, b):
    """
    Combine two dicts into a new dict, using the second dict's values for keys appearing in both dicts
    :param a: First dict
    :param b: Second dict
    :return: dictionary with combined key-value pairs
    """
    c = a.copy()
    try:
        for key, val in b.items():
            if type(val) == dict:
                c[key] = combine_dict(a[key], b[key])
            else:
                c[key] = val
    except AttributeError:  # In case other isn't a dict
        return NotImplemented

    return c
