[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10017540.svg)](https://doi.org/10.5281/zenodo.10017540)
[![Documentation Status](https://readthedocs.org/projects/smv2rho/badge/?version=latest)](https://smv2rho.readthedocs.io/en/latest/?badge=latest)


# SMV2rho

 `SMV2rho` is a Python package for converting continental crustal seismic velocities into density.
 It contains modules for doing the following:

 * [`density_functions`](src/SMV2rho/density_functions.py): for calculating density of continental crustal rocks given P- and S-wave velocity profiles using either Brocher's (2005) parametrisation, or a new scheme developed by Stephenson et al (_in review_; i.e. `SMV2rho`),
 * [`temperaure_dependence`](src/SMV2rho/temperature_dependence.py): for calculating the effects of temperature dependence on seismic velocity and density,
 * [`concident_profile_functions`](src/SMV2rho/coincident_profile_functions.py): for locating seismic velocity profiles that are within a certain distance of one another and comparing data,
 * [`spatial_functions`](src/SMV2rho/spatial_functions.py): for calculating distances between geographic locations,
 * [`uncertainties`](src/SMV2rho/uncertainties.py): for calculating uncertainties in crustal density estimates
 * [`plotting`](src/SMV2rho/plotting.py): for plotting velocity and density profiles, scatter plots to compare profiles, maps to view bullk properties and other useful plotting routines.


## Documentation

 Please access the documentation for this software here: [Documentation](https://smv2rho.readthedocs.io/en/latest/).


## Installation

First download the repository and save it on your machine.  The easiest way to do this is by opening a terminal and typing:

```
git clone https://github.com/sstephenson2/SMV2rho.git /path/to/your/directory
```

replacing `/path/to/your/directory` with the location that you would like to save the distribution.

Alternatively, you can simply click on the repository release and download it using the `GitHub` interface.  This way you can choose the release that you would like to download.

### Using conda

#### Unix

To create a new environment named "density" for running SMV2rho, follow these steps in your terminal:

1. Open your terminal.
2. Navigate to the directory where you have the SMV2rho package.
3. Run the following command to create a new conda environment named "density" and install the necessary dependencies:

```
conda create -n density python=3.12
```

After creating the environment, activate it with

```
conda activate density
```

### Using pip

#### Unix or Windows

If you're using pip, you can use `virtualenv` to create a new environment.  First install `virtualenv` if you haven't already:

```
pip install virtualenv
```

Then create a new environment called "density":

```
virtualenv density
```

To activate the environment, use:

* on windows:

```
density\scripts\activate
```

* on Unix or MacOS:

```
source density/bin/activate
```

### Installing SMV2rho

Once the `"density"` environment is activated, you can install `SMV2rho`. If you have a local copy of the `SMV2rho` repository, navigate to the root directory of the repository, where the `pyproject.toml` file is located. Then run the following command:

```
pip install -e .
```

This command installs the package in editable mode, which means you can modify the source code and see the effects without having to reinstall the package.  It will install the package in your default installation directory.

## Usage

If using this software please cite it in this way:

Stephenson, S. N., & Hoggard, M. J. (2024). SMV2rho (Version v1.0.0) [Computer software]. [DOI:10.5281/zenodo.10017541](https://doi.org/10.5281/zenodo.10017541)

`SMV2rho` is intended to be used for converting 1D velocity profiles into density.  We recommend
initially using it with [`SeisCruST`](https://github.com/sstephenson2/SeisCRUST), a global database of seismic continental crustal structure.  [`SeisCruST`](https://github.com/sstephenson2/SeisCRUST) can be accessed here https://github.com/sstephenson2/SeisCRUST.  Velocity profiles in `SeisCruST` are distributed in the correct format for use directly in `SMV2rho`.

Velocity profiles **must** be organised in the following way:

      ```
      profile_name
      lon lat
      moho_depth
      vs1 -z1
      vs2 -z2
      .    .
      .    .
      .    .
      ```

with depth in negative kilometres.

Users can choose their own file structure, but it is recommended to use a database structured in the following way.

      ```
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
      ```

This structure is given in the `TEST_DATA/` directory in this distribution.  This directory contains a couple of example velocity profile data files.

### Tutrorials

Please refer to the jupyter notebook tutorials in this distribution [`TUTORIAL`](TUTORIALS/).

Note that to run the tutorials you will need to install `jupyter`.  Installation of jupyter notebook is not automatic with the installation described above.  We recommend running the notebooks in an ide such as VSCode or Spyder.

  - [`tutorial1`](TUTORIALS/tutorial_1.ipynb): steps through how to load a velocity profile file into `SMV2rho`.
  - [`tutorial2`](TUTORIALS/tutorial_2.ipynb): demonstrates how to convert a velocity profile
  to density using Brocher's (2005) approach.
  - [`tutorial3`](TUTORIALS/tutorial_3.ipynb): further explores converting a $V_S$ profile into density using Brocher's (2005) approach.
  - [`tutorial4`](TUTORIALS/tutorial_4.ipynb): introduces the new `SMV2rho` density conversion scheme and explores converting to pressure-dependent density at s.t.p.
  - [`tutorial5`](TUTORIALS/tutorial_5.ipynb): demonstrates how to include temperature dependence in the density conversion to convert to density at depth rather than density at s.t.p.
  - [`tutorial6`](TUTORIALS/tutorial_6.ipynb): shows how to estimate uncertainties in the temperature-dependent `SMV2rho` density conversion scheme.
  - [`tutorial7`](TUTORIALS/tutorial_7.ipynb):  steps through how to...
    - clone the `SeisCruST` database,
    - convert multiple profiles in one batch,
    - estimate the relationship between bulk crustal density and crustal thickness,
    - and some basic ways to explore relationships between the profiles and calculate stastistics about the data.

## Contributing

We enthusiastically encourage contributions to `SMV2rho`.  Whether you have identified a bug, you think there is some functionality missing, or you have an idea for an improved mthod for calculating crustal density then please do getg in touch.

If you're ready to contribute, here's how you can do it:

1. **Fork the repository**: Click the 'Fork' button at the top right of this page and clone your forked repository to your local machine.

2. **Create a new branch**: From your terminal, create a new branch for the new method that you want to add, bug you want to fix, or feature you want to work on. You can create a new branch with `git checkout -b branch-name`.

3. **Make your changes**: Make the changes you want to contribute. Whether it is adding a new method to convert velocity to density, fixing a bug, improving documentation, or adding a new feature of some other type, we really appreciate feedback, new features and improvements!

4. **Commit your changes**: Once you're done, commit your changes with `git commit -m "Your detailed commit message"`.  Please add as much information as possible to your commit messages.

5. **Push your changes**: Push your changes to your forked repository with `git push origin branch-name`.

6. **Create a pull request**: Go to your forked repository on GitHub and click the 'New pull request' button. Fill out the form and then submit the pull request.

We will take a look at your pull request as soon as we can!

## License

This software is distributed under an MIT license.  Please read the [`LICENSE`](LICENSE) file for more information.  We just ask that this software package is cited in any work that uses `SMV2rho` and that the primary study for the relevant apprach is also cited.  For `SMV2rho` that is Steophenson _et al._ (2024).