from typing import List, Dict, Union

class CustomBasis:
    '''
    A class to represent a custom parameter basis.

    Parameters
    ----------
    name : str
        The name of the custom basis.
    parameters : list or dict
        The parameters in the custom basis. If a list is provided, all parameters are assumed to be real. If a dict is provided, the keys are the parameter names and the values are the types (e.g., 'R' for real, 'C' for complex).

    Attributes
    ----------
    name : str
        The name of the custom basis.
    parameters : dict
        The parameters in the custom basis. The keys are the parameter names and the values are the types (e.g., 'R' for real, 'C' for complex).

    Methods
    -------
    get_all_names() -> List[str]
        Get all custom basis names.
    get(name: str) -> 'CustomBasis'
        Get a custom basis by name.
    get_all() -> List['CustomBasis']
        Get all custom basis objects.
    get_parameter_basis() -> List
        Get the parameter basis.

    Examples
    --------
    Initialize a custom basis with real parameters:

    >>> custom_basis = CustomBasis('example_basis', ['param1', 'param2'])

    Initialize a custom basis with mixed parameters:

    >>> custom_basis = CustomBasis('example_basis', {'param1': 'R', 'param2': 'C'})

    Get all custom basis names:

    >>> CustomBasis.get_all_names()

    Get a custom basis by name:

    >>> custom_basis = CustomBasis.get('example_basis')

    Get all custom basis objects:

    >>> custom_bases = CustomBasis.get_all()

    Get the parameter basis:

    >>> parameter_basis = custom_basis.get_parameter_basis()

    '''

    _custom_bases: Dict[str, 'CustomBasis'] = {}  # Class attribute to store all custom bases

    def __init__(self, name: str, parameters: Union[List[str], Dict[str, str]]):
        """
        Initialize the CustomBasis class.

        Parameters
        ----------
            name : str
                The name of the custom basis.
            parameters : list or dict
                The parameters in the custom basis. If a list is provided, all parameters are assumed to be real. If a dict is provided, the keys are the parameter names and the values are the types (e.g., 'R' for real, 'C' for complex).

        Returns
        --------
            None

        Examples
        --------
        Initialize a custom basis with real parameters:

        >>> custom_basis = CustomBasis('example_basis', ['param1', 'param2'])

        Initialize a custom basis with mixed parameters:

        >>> custom_basis = CustomBasis('example_basis', {'param1': 'R', 'param2': 'C'})
        """
        self.name = name
        if isinstance(parameters, list):
            self.parameters = {param: 'R' for param in parameters}
        elif isinstance(parameters, dict):
            if not all(value in ['R', 'C'] for value in parameters.values()):
                raise ValueError("All parameter types must be either 'R' or 'C'.")
            self.parameters = parameters
        else:
            raise ValueError("Parameters must be a list or a dictionary.")
        self._custom_bases[self.name] = self

    @classmethod
    def get_all_names(cls) -> List[str]:
        """
        Get all custom basis names.

        Returns
        --------
            list
                A list of all custom basis names.

        Examples
        --------
            >>> CustomBasis.get_all_names()
        """
        return sorted(cls._custom_bases.keys())

    @classmethod
    def get(cls, name: str) -> 'CustomBasis':
        """
        Get a custom basis by name.

        Parameters
        ----------
            name : str
                The name of the custom basis.
        Returns
        --------
            CustomBasis
                The custom basis object.

        Examples
        --------
            >>> CustomBasis.get('example_basis')
        """
        return cls._custom_bases.get(name)

    @classmethod
    def get_all(cls) -> List['CustomBasis']:
        """
        Get all custom basis objects.

        Returns
        --------
            list
                A list of all custom basis objects.

        Examples
        --------
            >>> CustomBasis.get_all()
        """
        return list(cls._custom_bases.values())

    def get_parameter_basis(self, split_re_im=True) -> List:
        """
        Get the parameter basis.

        Parameters
        ----------
            split_re_im : bool, optional
                If True, split parameters into real and imaginary parts, otherwise return the parameters directly. Default is True.

        Returns
        --------
            list
                A list containing the parameter basis.

        Examples
        --------
            >>> custom_basis = CustomBasis('example_basis', ['param1', 'param2'])
            >>> custom_basis.get_parameter_basis('example_basis')
        """
        if split_re_im:
            parameter_basis = []
            for parameter, parameter_type in self.parameters.items():
                parameter_basis.append((parameter, 'R'))
                if parameter_type == 'C':
                    parameter_basis.append((parameter, 'I'))
        else:
            parameter_basis = self.parameters.keys()
        return sorted(parameter_basis)
