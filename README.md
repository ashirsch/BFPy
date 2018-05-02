# kemitter

### A Python environment for wide-angle energy-momentum spectroscopy

Read the documentation at [here](https://ashirsch.github.io/kemitter/).

### Installation
Currently you can install `kemitter` from source by cloning this repository. 

Some of the `cvxpy` dependencies can be tricky to install. On Windows we have encountered build issues with some of 
the `cvxpy` solvers, so if you encounter problems you may want to install `ECOS` from 
a [pre-built wheel](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ecos) before attempting to install `kemitter`.

```
git clone https://github.com/ashirsch/kemitter.git
pip install numpy
pip install -e .
```

Finally, follow the instructions on the [MOSEK website](https://docs.mosek.com/8.1/install/installation.html) to 
install the Python API and to obtain an academic license. Additional solver support will be available 
in the next release.

#### Version 1.0.0a