Usage
=====

`SMV2rho` is intended to be used for converting 1D velocity profiles into density.  We recommend
initially using it with `SeisCruST`_, a global database of seismic continental crustal structure.  `SeisCruST`_ can be accessed here https://github.com/sstephenson2/SeisCRUST.  Velocity profiles in `SeisCruST` are distributed in the correct format for use directly in `SMV2rho`.

Velocity profiles **must** be organised in the following way::

   profile_name
   lon lat
   moho_depth
   vs1 -z1
   vs2 -z2
   .    .
   .    .
   .    .

with depth in negative kilometres.

Users can choose their own file structure, but it is recommended to use a database structured in the following way::

   TEST_DATA
   |- TEST_DATA
   |  |- EUROPE
   |  |  `- Vp
   |  |     `- RECEIVER_FUNCTION
   |  |        `- DATA
   |  `- HUDSON_BAY
   |     `- Vs
   |        `- RECEIVER_FUNCTION
   |           `- DATA

This structure is given in the `TEST_DATA/` directory in this distribution.  This directory contains a couple of example velocity profile data files.

.. _SeisCruST: https://github.com/sstephenson2/SeisCRUST