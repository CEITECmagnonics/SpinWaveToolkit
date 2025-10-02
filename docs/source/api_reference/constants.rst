Constants
=========

.. currentmodule:: SpinWaveToolkit

.. tip::
    All of these are also accessible from the module level, e.g. ``SpinWaveToolkit.MU0``.

.. autodata:: SpinWaveToolkit.helpers.MU0

.. autodata:: SpinWaveToolkit.helpers.C

.. autodata:: SpinWaveToolkit.helpers.HBAR

.. autodata:: SpinWaveToolkit.helpers.KB

.. The constants have to be referenced to their true position within the module, since the doc-comments are extracted from this file. Using SpinWaveToolkit.MU0 would not get the doc-comment text.

.. Also, using the currentmodule directive does not work with SWT.helpers, since helpers is not a module (i.e. does not contain __init__.py), but rather a tree structure.

Pre-defined materials (constants of type :class:`~Material`) are defined in :doc:`classes/Material` class documentation.
