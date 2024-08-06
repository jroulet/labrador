import logging
import setuptools_scm


try:
    __version__ = setuptools_scm.get_version(root='..', relative_to=__file__)
except Exception:
    logging.warning(f'Could not determine {__name__} package version.')
    __version__ = None
