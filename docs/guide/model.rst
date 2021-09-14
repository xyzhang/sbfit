******
Models
******

We use ``astropy.modeling.Model`` as model base. Similarly, all model instances can be combined as a
``astropy.modeling.CompoundModel`` instance using the ``+`` operator.

Implemented models
==================

- :py:class:`~sbfit.model.Constant`
- :py:class:`~sbfit.model.Gaussian`
- :py:class:`~sbfit.model.DoublePowerLaw`
- :py:class:`~sbfit.model.Beta`
- :py:class:`~sbfit.model.ConeDoublePowerLaw`


User model
==========

For models that have not been implemented, it is easy to be defined using :py:func:`~sbfit.model.custom_model`
decorator. Here is a simple example to create an unprojected power law model yourself.

.. code-block:: python

   import sbfit.model

   @sbfit.model.custom_model
   def PowerLaw(x, norm=1, alpha=1):
       return norm * x ** -alpha

   my_pl = PowerLaw()

.. note:: The first argument of a user model should be ``x``. Model parameters are given from the second argument with
   default values. To follow the convention, the first parameter should always be the normalization ``norm``.
