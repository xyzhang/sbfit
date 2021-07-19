*******
Regions
*******

Region format
-------------

The ``sbfit`` package supports `DS9 <https://ds9.si.edu>`_ format region files.
Each region file should contain one :ref:`inclusive region<Inclusive regions>`
and any number of :ref:`exclusive regions<Exclusive regions>`.
Currently, only the ``image``, ``fk5``, and ``icrs`` frames are supported.

## Include/exclude property
Different from the original Include/Exclude property of the DS9 regions, the 
``sbfit`` package defines this property based on the type of the regions. The
description of each region type can be found at 
`<http://ds9.si.edu/doc/ref/region.html>`_.

Inclusive regions
-----------------

Inclusive regions are used to extract a surface brightness profile. They
are:

* ``panda``
* ``epanda``
* ``projection``

.. note:: For regions ``panda`` and ``epanda``, the numbers of angles and radii should be set to 1.



Exclusive regions
-----------------
Exclusive regions are used for masking sources. They are:

- ``circle``
- ``ellipse``