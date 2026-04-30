Functions
=========

.. currentmodule:: SpinWaveToolkit.bls

BLS signal calculation
----------------------

The following functions provide different approaches to compute the BLS signal, either using the Green function formalism or the Reciprocity Theorem. Both give similar results and the reader is redirected to the source publications for further explanation:

Green function method:
    Wojewoda et al., *Physical Review B* **110**, 224428 (2024). DOI: `10.1103/PhysRevB.110.224428 <https://doi.org/10.1103/PhysRevB.110.224428>`_

Reciprocity Theorem method:
    Krčma et al., *Science Advances* **11**, eady8833 (2025). DOI: `10.1126/sciadv.ady8833 <https://doi.org/10.1126/sciadv.ady8833>`_

The Green function method (:func:`~get_signal_GF_focal` function) takes into account more optical effects happening at the sample surface and can compute signal from below a multilayer material stack, while the Reciprocity Theorem method (``get_signal_RT...`` functions) is more straightforward and computationally less expensive.

Currently, the fastest approach is via the Reciprocity Theorem with electric fields input in the reciprocal space (:func:`~get_signal_RT_pupil`), which is the **recommended approach** for most cases. The other functions are suitable for comparison and validation purposes. The ``get_signal_...`` functions use the electic fields given by the respective method of the :class:`~ObjectiveLens` class, i.e. :py:meth:`GetFocalField` or :meth:`GetPupilField` for the electric field in the real or reciprocal space, respectively. (The correct space is also indicated by the suffix of the ``get_signal_...`` function name.)

|

.. autofunction:: get_signal_GF_focal

.. autofunction:: getBLSsignal

.. autofunction:: get_signal_RT_pupil

.. autofunction:: get_signal_RT_focal

.. autofunction:: get_signal_RT_focal_3d


Helper functions
----------------

Functions used internally for the BLS signal calculations, especially using the Green function method, but may also be useful for other purposes.

.. autofunction:: fresnel_coefficients

.. autofunction:: sph_green_function



