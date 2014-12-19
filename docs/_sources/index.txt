.. chemreac documentation master file, created by
   sphinx-quickstart on Tue Aug 26 17:03:58 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to chemreac's documentation!
====================================

chemreac is an open-source library for modeling chemical kinetcs in either:

- Homogeneous bulk solution (no concentration gradients)
    - Kinetics governed by law of mass action
- Non-homogeneous solution where concentration profile has either:
    - planar symmetry
    - cylindrical symmetry
    - spherical symmetry

For the non-homogeneous case the user may choose:

- reflective or interpolating boundary conditions
- number of stencil points (3, 5 or 7)
- an arbitrarily spaced grid
- to calculate the electric field from concentrations for advection (drift).

Interfaces are provided to Sundials_ (CVODE) at the C++ level and to both 
Sundials and SciPy_ at the python level.

Currently the code is written with the following assumptions:

- isothermal conditions
- law of mass action kinetics
- low concentrations of charged species


The Advection/Diffusion/Reaction model
--------------------------------------

The model is formulated as the Smoluchovski equation:

.. math ::

    \frac{\partial c_i}{\partial t} &= D_i \nabla^2c_i - \mu_i\vec{E}(\vec{c}) \cdot \nabla c_i + \sum_l r_l S_{il}

where :math:`t` is time, :math:`c_i` is the concentration of species :math:`i`, :math:`D_i` is the diffusion coefficient of the same species, :math:`\mu_i` is the electric mobility, :math:`\vec{E}` is the electric field, :math:`S_{il}` is the net stoichiometric coefficient of species :math:`i` in reaction :math:`l` and :math:`r_l` is the rate of reaction :math:`l`, which for a mass-action type of rate law is given by:

.. math ::

    r_l = \begin{cases} \kappa_l\prod_k c_k^{R_{kl}} &\mbox{if } \sum_k R_{kl} > 0 \\
    0 &\mbox{otherwise} \end{cases}

where :math:`\kappa_l` is the rate constant, :math:`R_{kl}` is the stoichiometric coefficient of species :math:`k` on the reactant side.

The equation is discretized in one dimension (flat, cylindrical or spherical shells). 
The model is implemented in a C++ class with Python_ bindings for ease of access.

Contents:

.. toctree::
   :maxdepth: 4

   install.rst
   modules.rst
   examples/index.rst
   dev.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _Python: https://www.python.org/
.. _Sundials: http://computation.llnl.gov/casc/sundials/main.html
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html#scipy.integrate.ode
.. _VODE: http://www.netlib.org/ode/vode.f
