.. SpinWaveToolkit documentation master file, created by
   sphinx-quickstart on Tue Aug 12 12:25:37 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SpinWaveToolkit Documentation
=============================



.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   getting_started
   examples
   user_guide
   api_reference/index
   release_notes
   publications


SpinWaveToolkit is an open-source Python package which provides analytical tools for spin-wave physics and research. It consist of several models used in everyday calculations as well as of special tools for more advanced spin-wave modelling.


Features
--------

* Calculation of the dispersion relation and derived quantities for several systems using analytical, semi-analytical, and numerical models. These include

  * single magnetic layer (thin film) surrounded by dielectrics,
  * coupled magnetic double layer (e.g. a synthetic antiferromagnet),
  * single magnetic layer inductively coupled to a superconducting layer from one side.

* Simple magnetic material management using a :py:class:`.Material` class.
* Functions for modelling Brillouin light scattering (BLS) signal and experiments.


Where to go
-----------

.. grid:: 3
    :gutter: 3

    .. grid-item-card::  :octicon:`rocket;6em;sd-text-primary`
        :text-align: center
        :link: getting_started
        :link-type: doc
        :link-alt: Getting Started
        :class-card: hover-card

        First time? Jump to **Getting Started**.

    .. grid-item-card::  :octicon:`graph;6em;sd-text-primary`
        :text-align: center
        :link: https://ceitecmagnonics.github.io/SWTweb/
        :link-alt: SWTweb
        :class-card: hover-card

        **SWTweb**: A fast in-browser calculator using SpinWaveToolkit — powerful, no installation, safe.

    .. grid-item-card::  :octicon:`image;4em;sd-text-primary` :octicon:`image;6em;sd-text-primary` :octicon:`image;4em;sd-text-primary`
        :text-align: center
        :link: examples
        :link-type: doc
        :link-alt: Examples
        :class-card: hover-card

        Get inspiration from the provided **Examples**.

    .. grid-item-card:: :octicon:`book;6em;sd-text-primary`
        :text-align: center
        :link: user_guide
        :link-type: doc
        :link-alt: User Guide
        :class-card: hover-card

        More information on the usage of SpinWaveToolkit is given in the **User Guide**.

    .. grid-item-card:: :octicon:`log;6em;sd-text-primary`
        :text-align: center
        :link: api_reference/index
        :link-type: doc
        :link-alt: API Reference
        :class-card: hover-card

        Details about a specific element can be found in the **API Reference**.

    .. grid-item-card:: :octicon:`mortar-board;6em;sd-text-primary`
        :text-align: center
        :link: publications
        :link-type: doc
        :link-alt: Publications
        :class-card: hover-card

        See **Publications** for published outputs and used literature.




.. _howtocite:

Cite us
-------

If you use SpinWaveToolkit in your work, please cite it as follows:

.. [1] Klíma, J. *et al.* "SpinWaveToolkit: Python package for (semi-)analytical calculations in the field of spin-wave physics" (2026) `arXiv:2601.23227 [cond-mat.mes-hall] <https://doi.org/10.48550/arXiv.2601.23227>`_
.. [2] Klíma, J., Krčma, J., & Wojewoda, O. "SpinWaveToolkit: Set of tools useful in spin wave research." GitHub, 2026. `<https://github.com/CEITECmagnonics/SpinWaveToolkit>`_


.. code-block:: bibtex

   @misc{klima2026,
       title={{SpinWaveToolkit}: {Python} package for (semi-)analytical calculations in the field of spin-wave physics},
       author={Klíma, Jan and Wojewoda, Ondřej and Krčma, Jakub and Hrtoň, Martin and Pavelka, Dominik and Holobrádek, Jakub and Urbánek, Michal},
       year={2026},
       eprint={2601.23227},
       archivePrefix={arXiv},
       primaryClass={cond-mat.mes-hall},
       url={https://arxiv.org/abs/2601.23227},
   }

   @online{swt,
       author = {Klíma, Jan and Krčma, Jakub and Wojewoda, Ondřej},
       title = {SpinWaveToolkit: Set of tools useful in spin wave research},
       year = {2026},
       publisher = {GitHub},
       version = {1.3.0},
       url = {https://github.com/CEITECmagnonics/SpinWaveToolkit},
       language = {en},
   }


All sources of models used within the module are cited in their respective documentation. Consider citing them as well if you use these models.
