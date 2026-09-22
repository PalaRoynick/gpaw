.. _densitymix:

==============
Density Mixing
==============


Pulay Mixing
------------

A simple and efficient method for mixing is the Pulay mixer [#Pulay1980]_,
[#Kresse1996]_.

Pulay mixing (or direct inversion of the iterative subspace (DIIS))
attempts to find a good approximation of the final solution as a
linear combination of a set of trial vectors `\{n^i\}` generated during
an iterative solution of a problem. If the error associated with a
given solution is given as `\{R^i\}` then Pulay mixing assumes that
the error of a linear combination of the trail vectors is given as the
same linear combination of errors

.. math::

  n_{i+1}=\sum \alpha_i n_i \quad,\quad R_{i+1}=\sum \alpha_i R_i

The norm `R^{i+1}` is thus given as

.. math::

  \langle R_{i+1}|R_{i+1}\rangle=\bar{\alpha}^T \bar{\bar{R}}\bar{\alpha}

where elements of the matrix is given as `\bar{\bar{R}}_{ij}=\langle
R_{i}|R_{j}\rangle`. The norm can thus be minimized by solving

.. math::

  \frac{\delta \langle R_{i+1}|R_{i+1}\rangle}{\delta
  \bar{\alpha}^T}=2 \bar{\bar{R}}\bar{\alpha}=0

In density mixing the error of a given input density is given as

.. math::

  R_i = n_i^{out}[n_i^{in}]-n_i^{in}

The original Pulay mixing only uses `n_i^{out}` to calculate the
errors and thereby the mixing parameters. To more efficiently cover
solution space it can be an advantage to include them with a certain
weight, given as the input parameter `\beta`.

.. math::

  n_{i+1}^{in}=\sum \alpha_i (n_i^{in}+\beta R_i)

In GPAW the Pulay mixer can be specified and configured by setting::

  mixer = {'name': 'pulay',  # The choice of mixer.
           'nmaxold': 16,  # Number of old densities to store.
           'beta': 0.08,  # The linear mixing parameter.
           'weight': 20,  # Scaling of the residual metric.
           'sigma': 0.015,  # Width of the residual metric.
           'g_ss': None}  # Experimental spin metric.

Residual Metric
---------------

Convergence can be improved by an optimized metric `\hat{M}` for
calculation of scalar products involving residuals, `\langle A | B
\rangle _s = \langle A | \hat{M} | B \rangle`, where `\langle \rangle
_s` is the scalar product with the special metric and `\langle
\rangle` is the usual scalar product.  The metric is based on the
rationale that contributions for small wave vectors are more important
than contributions for large wave vectors [#Kresse1996]_.  Using a
metric that weighs short wave density changes more than long wave
changes can reduce charge sloshing significantly.

It has been found [#Kresse1996]_ that the metric

.. math::

  \hat{M} = \sum_q | q \rangle f_q \langle q |, \quad f_q =
  1 + \frac{w}{q^2}

is particularly useful (`w` is a suitably chosen weight). However, as it
diverges for `q \to 0`, it is not applicable to quantities that sum to a finite
value, such as the magnetization. To mitigate this, we apply a modified version
of the metric

.. math::

  f_q = \frac{w + q^2}{\sigma + q^2}

With `\sigma` being a small regaularization/broadening term. We note that, in
general, the metric is only applied to the charge density unless otherwise
specified.

Advanced mixing: The MSR1 algorithm
-----------------------------------

The default mixer in GPAW is the MSR1 mixer [#MSR1]_ which promises faster and
more stable convergence compared to the Pulay mixer. It achieves this by
utilzing a dynamic balance of good Broyden (greedy) and bad Broyden (Pulay-like)
weighting for its step direction. Additionally, it also predicts its own step
sizes to ensure optimal convergence.

The default parameters used in GPAW are::

    mixer = {'name': 'msr1',  # The choice of mixer
             'beta': 0.05,  # Size of the Pratt step
             'nmaxold': 10,  # Number of previous steps used in mixing
             'reg': 0.005,  # Regularization (i.e. damping of the SVD inversion)
             'gb_scale': 1.0,  # Scaling of the amount of good-broyden used.
             'max_A': 0.75,  # Largest relative step taken along the residual.
             'trust_scale': 1.0,  # Scaling of the trust region radius.
             'soft_lim': 4.0,  # Error threshold for backstepping.
             'hard_lim': 6.0,  # Error threshold for history removal.
             'weight': 20,  # Scaling of the residual metric.
             'sigma': 0.015,  # Width of the residual metric
             'g_ss': None}  # Experimental spin metric.

These parameters should work well for most systems.

Mixing the magnetization separately: Multimixer
-----------------------------------------------

The most common example of the using the multi-mixer is to separately mix the
density and magnetization, this can be done by specifying the following::

    mixer = {'name': 'multi',  # The choice of mixer
             'method': 'mag-dens',  # Mix the density and magnetization separately
             'mixer_a': {...},  # The density mixer settings, e.g. MSR1
             'mixer_b': {...}}  # The magnetization mixer settings.

For some systems, this will improve convergence, while for others it will hinder
the convergence. When using the multi-mixer like this, it is recommended to
reduce the metric ``weight`` (or turn it off) and increase the ``beta`` value for
``mixer_b`` (the magnetization mixer).

Alternative uses of the multimixer is to set ``method`` to ``separate``, which
separately mixes the two spin channels. Generally speaking, this does not yield
improvements to convergence, though.

Specifying mixer through the corresponding parameter classes
------------------------------------------------------------

You can also use the mixer classes directly instead of the ``mixer``
dictionary, similar to other GPAW inputs::

    from gpaw.dft import BaseMixer, Pulay, MSR1, MultiMixer

These classes have the same parameters as corresponding ``mixer`` dictionary,
and can be helpful when using an IDE with autocompletion.

References
----------

.. [#Pulay1980] Pulay, Chem. Phys. Let. **73**, 393 (1980)
.. [#Kresse1996] Kresse, Phys. Rev. B **54**, 11169 (1996)
.. [#MSR1] Marks, J. Chem. Theory Comput. **17**, 5715 (2021)
