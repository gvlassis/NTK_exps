import datasets

def string_to_dataset_class(string):
    return getattr(datasets, string)

# Colors
GREEN = '#388E3C'
LIGHT_GREEN = '#8BC34A'
INDIGO = '#3F51B5'
BLUE = '#2196F3'