"""Input and output to HDF5."""
import importlib
import inspect
import numpy as np

import h5py

from . import __version__


def read_hdf5(file_path):
    """
    Read a file saved with ``HDF5Mixin.to_hdf5()``.

    Parameters
    ----------
    file_path : os.PathLike
        Path to file that we wish to load.

    Returns
    -------
    HDF5Mixin : Instance of (subclass of) ``HDF5Mixin``.
    """
    dic = load_dict_from_hdf5(file_path)
    return HDF5Mixin.read_object(dic)


class HDF5Mixin:
    """
    Provide HDF5 output to subclasses.

    Use `.to_hdf5()` for saving, `read_hdf5()` for loading.

    This mixin defines a method `.get_init_dict()` which works for
    classes that store their init parameters as attributes with the same
    names. If this is not the case, the subclass should override
    `.get_init_dict()`.
    """
    subclass_registry = {}

    @classmethod
    def read_object(cls, obj):
        """
        Intercept dictionaries created by ``HDF5Mixin.to_dict()``, and
        turn them into instances of the correct subclass.
        """
        if isinstance(obj, dict):
            if '__HDF5Mixin_subclass__' in obj:
                importlib.import_module(obj['__module__'])
                subclass = cls.subclass_registry[obj['__HDF5Mixin_subclass__']]
                init_kwargs = cls.read_object(obj['init_kwargs'])
                return subclass(**init_kwargs)
            return {key: cls.read_object(val) for key, val in obj.items()}

        if isinstance(obj, str) and obj == '__None__':
            return None

        return obj

    def __init_subclass__(cls):
        """Register subclasses."""
        super().__init_subclass__()
        HDF5Mixin.subclass_registry[cls.__name__] = cls

    def to_hdf5(self, file_path):
        """
        Save a dictionary to an HDF5 file.

        The file can be loaded with `read_hdf5()`.
        """
        save_dict_to_hdf5(file_path, self.to_dict())

    def get_init_dict(self, **kwargs):
        """
        Return dictionary with keyword arguments to `__init__`.

        Only works if the class stores its init parameters as attributes
        with the same names. Otherwise, the subclass should override
        this method.

        Parameters
        ----------
        **kwargs
            Allows to manually override some keys. The remaining
            ones will be read from the instance's attributes. All
            keywords must be in the __init__ signature. It's mostly
            here to facilitate overriding by subclasses.
        """
        keys = inspect.signature(self.__init__).parameters.keys()

        if extra_keys := kwargs.keys() - keys:
            raise ValueError(f'Extraneous keys {extra_keys}')

        try:
            init_dict = kwargs | {key: getattr(self, key)
                                  for key in keys - kwargs}
        except KeyError as err:
            raise KeyError(
                f'`{self.__class__.__name__}` must override `get_init_dict` '
                '(or store its init parameters with the same names).'
            ) from err

        for key, val in list(init_dict.items()):
            if isinstance(val, HDF5Mixin):
                init_dict[key] = val.to_dict()
            elif val is None:
                init_dict[key] = '__None__'

        return init_dict

    def to_dict(self):
        """
        Return special dictionary with information to reconstruct self.

        See Also
        --------
        .read_object() : Inverse of this function
        """
        return {'__HDF5Mixin_subclass__': self.__class__.__name__,
                '__module__': self._get_module_name(),
                '__version__': __version__,
                'init_kwargs': self.get_init_dict()}

    def _get_module_name(self):
        """Name of the module that defines the instance's class."""
        module = self.__class__.__module__
        if module == '__main__' and (spec := inspect.getmodule(self).__spec__):
            module = spec.name
        return module


def save_dict_to_hdf5(file_path, dic):
    """Save a dictionary to an HDF5 file."""
    with h5py.File(file_path, 'w') as hdf5_file:
        _serialize_to_hdf5(hdf5_file, dic)


def _serialize_to_hdf5(hdf5_group, dic):
    """Recursively store dictionary data in HDF5."""
    for key, value in dic.items():
        if isinstance(value, np.ndarray):
            hdf5_group.create_dataset(key, data=value)
        elif isinstance(value, dict):
            subgroup = hdf5_group.create_group(key)
            _serialize_to_hdf5(subgroup, value)
        else:
            try:
                hdf5_group.attrs[key] = value
            except TypeError as err:
                raise TypeError(f'Error trying to save {key}={value!r}'
                               ) from err

def load_dict_from_hdf5(file_path):
    """Load a dictionary from an HDF5 file."""
    with h5py.File(file_path, 'r') as hdf5_file:
        return _deserialize_from_hdf5(hdf5_file)


def _deserialize_from_hdf5(hdf5_group):
    """Recursively reconstruct dictionary data from HDF5."""
    dic = {}
    for key, item in hdf5_group.items():
        if isinstance(item, h5py.Dataset):
            dic[key] = item[()]
        else:
            dic[key] = _deserialize_from_hdf5(item)

    for key, value in hdf5_group.attrs.items():
        dic[key] = value

    return dic
