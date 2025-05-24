from typing import Iterable
import numpy as np
import re
import hashlib


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
        schema = json_data['$schema']
        if isinstance(schema, (np.ndarray, list)):
            schema = str(schema[0])
        else:
            schema = str(schema)
        match = json_schema_name_pattern.search(schema)
        if match:
            schema_name = match.group(1)
            schema_version = match.group(3)
    return schema_name, schema_version

def format_number(x):
    x = float(x)
    formatted = f"{x:.6f}"
    float_string = formatted.rstrip('0') if '.' in formatted else formatted
    if float_string.endswith('.'):
        float_string += '0'
    return float_string

def get_observable_key(name_args):
    if isinstance(name_args, str):
        return name_args
    if len(name_args) == 1:
        return name_args[0]
    name = name_args[0]
    args = name_args[1:]
    return f"('{name}', " + ", ".join(format_number(x) for x in args) + ")"

def escape(name: str) -> str:
    return name.replace('\\', '\\\\').replace('|', '\\|')

def hash_observable_names(row_names: Iterable[str], col_names: Iterable[str]) -> str:
    row_escaped = '|'.join(escape(o) for o in sorted(row_names))
    col_escaped = '|'.join(escape(o) for o in sorted(col_names))
    block_id = row_escaped + '||' + col_escaped
    return hashlib.md5(block_id.encode('utf-8')).hexdigest()
