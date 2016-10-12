from chainplot.utils.dict_tools import replace_dict, split_kwargs, britishdict, combine_dict
from chainplot.utils.string_tools import prettify


class Mapping:
    def __init__(self):
        self.aes = {}

    def update_mapping(self, **kwargs):
        self.aes = combine_dict(self.aes.copy(), kwargs)

    def pull_data(self, attr, data):
        data = data.copy()

        if type(attr) == str:
            data[self.aes[attr]]
        elif callable(attr):
            attr(data)

        return


