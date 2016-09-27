def replace_dict(input_dict, val=None, replacement=''):
    output_dict = input_dict.copy()
    for k, v in output_dict.items():
        if v is val:
            output_dict[k] = replacement
    return output_dict


def split_kwargs(kwargs_dict, prefix='shadow_'):
    input_dict = kwargs_dict.copy()
    output_dict = {}
    shadow_dict = {}

    for k, v in input_dict.items():
        if bool(re.match(prefix + '.*', k)):
            shadow_k = re.sub(prefix, '', k)
            shadow_dict[shadow_k] = v
        else:
            output_dict[k] = v

    return output_dict, shadow_dict


def to_kwargs(**kwargs):
    return kwargs


def britishdict(argdict):
    outdict = argdict.copy()

    # Fix awful americanisms
    if 'colour' in outdict.keys():
        outdict['color'] = outdict['colour']
        outdict.pop('colour', 0)
    if 'edgecolour' in outdict.keys():
        outdict['edgecolor'] = outdict['edgecolour']
        outdict.pop('edgecolour', 0)

    return outdict


def combine_dict(a, b):
    c = a.copy()
    try:
        for key, val in b.items():
            if type(val) == dict:
                c[key] = combine_dict(a[key], b[key])
            else:
                c[key] = val
    except AttributeError:  # In case oth isn't a dict
        return NotImplemented

    return c
