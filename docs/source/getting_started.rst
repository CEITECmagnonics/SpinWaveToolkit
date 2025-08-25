Getting Started
===============

.. warning::
    
   Needs to be updated!


Installation
------------

The easiest way to install `SpinWaveToolkit` is via pip from PyPI. To do this, open the command line and type

.. code-block:: bash

    py -m pip install SpinWaveToolkit --user

Other installation approaches are descried in the :doc:`User Guide <user_guide/usage/installation>`.


Setting up the experiment
-------------------------
Import needed packages. For plotting, we will use `matplotlib`, but `SpinWaveToolkit` does not depend on it.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   import SpinWaveToolkit as swt

Choose a model
^^^^^^^^^^^^^^
First, you need to choose the appropriate model for your experiment. This depends mainly on the configuration. Currently, there are these dispersion models:

- single magnetic layer (zeroth perturbation - omits intermode coupling, mainly for thin films) - :py:class:`SingleLayer`
- single magnetic layer with intermode coupling (useful for thicker layers) - :py:class:`SingleLayerNumeric`
- two coupled magnetic layers (e.g. syntheric antiferromagnets) - :py:class:`DoubleLayerNumeric`
- one magnetic layer dipolarly coupled to a superconducting layer - :py:class:`SingleLayerSCcoupled`
- magnon-polariton in a bulk ferromagnet (very small wavevectors) - :py:class:`BulkPolariton`

Let's assume a single magnetic layer for the following examples. Therefore, we will use the :py:class:`SingleLayer` class.

Define your material
^^^^^^^^^^^^^^^^^^^^
To handle materials, `SpinWaveToolkit` uses the :py:class:`Material` class. You can either use one of the predefined materials (see documentation of :py:class:`Material`), or define your own by specifying its parameters

.. code-block:: python

   NiFe = swt.Material(Ms=800e3, A=16e-12, alpha=0.007, gamma=30*2e9*np.pi)


Set up geometry and conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   Bext = 10e-3  # (T) magnetic field
   d = 30e-9  # (m) thickness of the layer
   k = np.linspace(0, 30e6, 200)  # (rad/m) wavevector range
   theta = np.pi/2  # (rad) angle of magnetization from thin film normal
   phi = np.pi/2  # (rad) angle of wavevector from in-plane magnetization
   bc = 1  # boundary condition (1 for totally unpinned)

   # initialize the model
   sl = swt.SingleLayer(Bext, NiFe, d, k, theta, phi, boundary_cond=bc)


Retrieve dispersion relation
----------------------------



Calculate other quantities
--------------------------


Change parameters
-----------------


Sweeps
^^^^^^








