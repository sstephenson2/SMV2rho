Installation
============

First download the repository and save it on your machine.  The easiest way to do this is by opening a terminal and typing::

   git clone https://github.com/sstephenson2/SMV2rho.git /path/to/your/directory

replacing ``/path/to/your/directory`` with the location that you would like to save the distribution.

Alternatively, you can simply click on the repository release and download it using the `GitHub` interface.  This way you can choose the release that you would like to download.

Using conda
-----------

Unix
^^^^

To create a new environment named "density" for running SMV2rho, follow these steps in your terminal:

1. Open your terminal.
2. Navigate to the directory where you have the SMV2rho package.
3. Run the following command to create a new conda environment named "density" and install the necessary dependencies::

   conda create -n density python=3.12

After creating the environment, activate it with::

   conda activate density

Using pip
---------

Unix or Windows
^^^^^^^^^^^^^^^

If you're using pip, you can use `virtualenv` to create a new environment.  First install `virtualenv` if you haven't already::

   pip install virtualenv

Then create a new environment called "density"::

   virtualenv density

To activate the environment, use:

* on windows::

   density\scripts\activate

* on Unix or MacOS::

   source density/bin/activate

Installing SMV2rho
------------------

Once the ``"density"`` environment is activated, you can install `SMV2rho`. If you have a local copy of the `SMV2rho` repository, navigate to the root directory of the repository, where the `pyproject.toml` file is located. Then run the following command::

   pip install -e .

This command installs the package in editable mode, which means you can modify the source code and see the effects without having to reinstall the package.  It will install the package in your default installation directory.

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