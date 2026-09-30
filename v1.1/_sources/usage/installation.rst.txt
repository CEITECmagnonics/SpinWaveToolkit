Installation
============

Currently you can either 

1. *(recommended)* install latest release from PyPI via `pip` by typing in the command line

.. code-block:: bash

    py -m pip install SpinWaveToolkit --user

2. or install from GitHub any branch via `pip` by typing in the command line

.. code-block:: bash

    py -m pip install https://github.com/CEITECmagnonics/SpinWaveToolkit/tarball/<branch-name> --user

.. collapse:: older installation approaches *(not recommended)*

    3. or copy the `SpinWaveToolkit` folder to your ``site-packages`` folder manually. Usually (on Windows machines) located at

    .. code-block::

        C:\Users\<user>\AppData\Roaming\Python\Python<python-version>\site-packages

    for user-installed modules, or at 

    .. code-block::
        
        C:\<python-installation-folder>\Python<python-version>\Lib\site-packages

    for global modules.

    .. note::

        This approach still works, since `SpinWaveToolkit` is a pure Python package without any compiled extensions. However, it is not recommended, since it does not automatically install dependencies.


Dependencies
------------

The SpinWaveToolkit package is compatible with Python >3.7, and uses the following modules:

- :py:mod:`numpy` >1.20 (>2.0 is also ok, bugs be reported in `Issues <https://github.com/CEITECmagnonics/SpinWaveToolkit/issues>`_)
- :py:mod:`scipy` >1.8

.. note::

   If you encounter compatibility errors in contradiction with this list, let us know by posting your findings in a new `Issue <https://github.com/CEITECmagnonics/SpinWaveToolkit/issues>`_.
