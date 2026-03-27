"""Input and output to HDF5."""
import importlib
import numpy as np

import h5py
import cogwheel.utils

from . import __version__


def read_hdf5(file_path):
    """
    Read a file saved with :py:meth:`HDF5Mixin.to_hdf5`.

    Parameters
    ----------
    file_path : os.PathLike
        Path to file that we wish to load.

    Returns
    -------
    HDF5Mixin : Instance of (subclass of) ``HDF5Mixin``.
    """
    with h5py.File(file_path, 'r') as hdf5_file:
        return _deserialize_from_hdf5(hdf5_file)


class HDF5Mixin(cogwheel.utils.InitDictMixin):
    """
    Provide HDF5 output to subclasses.

    Use `.to_hdf5()` for saving, `read_hdf5()` for loading.

    This mixin defines a method `.get_init_dict()` which works for
    classes that store their init parameters as attributes with the same
    names. If this is not the case, the subclass should override
    `.get_init_dict()`.
    """
    subclass_registry = {}

    def __init_subclass__(cls):
        """Register subclasses."""
        super().__init_subclass__()
        HDF5Mixin.subclass_registry[cls.__name__] = cls

    def to_hdf5(self, file_path):
        """
        Save a dictionary to an HDF5 file.

        The file can be loaded with :py:func:`read_hdf5`.
        """
        with h5py.File(file_path, 'w') as hdf5_file:
            _serialize_to_hdf5(hdf5_file, self.to_dict())

    def to_dict(self):
        """
        Return special dictionary with information to reconstruct self.
        """
        return {'__HDF5Mixin_subclass__': self.__class__.__name__,
                '__module__': self.get_module_name(),
                '__version__': __version__,
                'init_kwargs': self.get_init_dict()}


def _serialize_to_hdf5(hdf5_group, dic):
    """Recursively store dictionary data in HDF5."""
    for key, value in dic.items():
        if isinstance(value, np.ndarray):
            hdf5_group.create_dataset(key, data=value)
        elif isinstance(value, dict):
            subgroup = hdf5_group.create_group(key)
            _serialize_to_hdf5(subgroup, value)
        elif value is None:
            hdf5_group.attrs[key] = '__None__'
        elif isinstance(value, HDF5Mixin):
            subgroup = hdf5_group.create_group(key)
            _serialize_to_hdf5(subgroup, value.to_dict())
        else:
            try:
                hdf5_group.attrs[key] = value
            except TypeError as err:
                raise TypeError(f'Error trying to save {key}={value!r}'
                               ) from err


def _deserialize_from_hdf5(hdf5_group):
    """Recursively reconstruct dictionary data from HDF5."""
    dic = {}
    for key, item in hdf5_group.items():
        if isinstance(item, h5py.Dataset):
            dic[key] = item[()]
        else:
            dic[key] = _deserialize_from_hdf5(item)

    for key, value in hdf5_group.attrs.items():
        if isinstance(value, str) and value == '__None__':
            dic[key] = None
        else:
            dic[key] = value

    if "__HDF5Mixin_subclass__" in dic:
        importlib.import_module(dic["__module__"])
        cls = HDF5Mixin.subclass_registry[dic["__HDF5Mixin_subclass__"]]
        return cls(**dic["init_kwargs"])

    return dic
