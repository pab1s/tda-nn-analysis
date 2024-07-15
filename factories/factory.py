from typing import Any

class Factory:
    """
    A class that represents a factory for creating objects based on a given key.

    Attributes:
        _creators (dict): A dictionary that maps keys to object creators.

    Methods:
        register(key, creator): Registers a creator function for a given key.
        create(key, **kwargs): Creates an object using the registered creator function for the given key.

    """

    def __init__(self):
        self._creators = {}

    def register(self, key, creator) -> None:
        """
        Registers a creator function for a given key.

        Args:
            key (str): The key associated with the creator function.
            creator (function): The function that creates the object.

        """
        self._creators[key] = creator

    def create(self, key, **kwargs) -> Any:
        """
        Creates an object using the registered creator function for the given key.

        Args:
            key (str): The key associated with the creator function.
            **kwargs: Additional keyword arguments to be passed to the creator function.

        Returns:
            object: The created object.

        Raises:
            ValueError: If the given key is not registered.

        """
        creator = self._creators.get(key)
        if not creator:
            raise ValueError(f"Unknown configuration: {key}")
        return creator(**kwargs)
