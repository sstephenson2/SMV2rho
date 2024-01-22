# SMV2rho

 SMV2rho is a Python package for converting continental crustal seismic velocities into density.
 It contains modules for doing the following:

 * calculating densities of continental crustal rocks given P- and S-wave velocity profiles using either Brocher's (2005) parametrisation, or a new scheme developed by Stephenson et al (_in review_)
 * calculating the effects of temperature dependence on seismic velocity and density
 * locating seismic velocity profiles that are within a certain distance of one another and comparing data
 * calculating distances between geographic locations
 * calculating uncertainties in crustal density estimates


## Installation

### Unix

##### Using conda

To create a new environment named "density" for running SMV2rho, follow these steps in your terminal:

1. Open your terminal.
2. Navigate to the directory where you have the SMV2rho package.
3. Run the following command to create a new conda environment named "density" and install the necessary dependencies:

```conda create -n density python=3.12```

After creating the environment, activate it with

```conda activate density```

### Unix or Windows

##### Using pip

If you're using pip, you can use `virtualenv` to create a new environment.  First install `virtualenv` if you haven't already:

```pip install virtualenv```

Then create a new environment called "density":

```virtualenv density```

To activate the environment, use:

* on windows:

```density\scripts\activate```

* on Unix or MacOS:

```source density/bin/activate```

