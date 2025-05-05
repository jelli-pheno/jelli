import numpy as np
import re


# Function to pad arrays to the same length repeating the last element
def pad_arrays(arrays):
    max_len = max(len(arr) for arr in arrays)
    return np.array([
        np.pad(arr, (0, max_len - len(arr)), mode='edge')
        for arr in arrays
    ])

json_schema_name_pattern = re.compile(
    r"/([a-zA-Z0-9_-]+?)(-(\d+(\.\d+)*))?(\.[a-zA-Z0-9]+)*$"
)
def get_json_schema(json_data):
    '''
    Extract the schema name and version from the JSON data.

    Parameters
    ----------
    json_data : dict
        The JSON data containing the schema information.

    Returns
    -------
    tuple
        A tuple containing the schema name and version. If not found, returns (None, None).
    '''
    schema_name = None
    schema_version = None
    if '$schema' in json_data:
        match = json_schema_name_pattern.search(json_data['$schema'])
        if match:
            schema_name = match.group(1)
            schema_version = match.group(3)
    return schema_name, schema_version
