SParse LINear INVersion (splinv)

A code to inverse linear transform using sparse regularization.

---
# Install

```shell
git clone https://github.com/mr-superonion/splinv.git
pip install -e .
```

# darkmapper

Reconstruct dark matter mass map from weak gravitational lensing shear
measurements from background galaxies.

+ [paper 1](https://ui.adsabs.harvard.edu/abs/2021ApJ...916...67L/abstract):
    + [demo1](demos/demo1.py) simulates shear distortion field on pixel grids
      caused by a NFW halo.
    + [demo2](demos/demo2.py) reconstruct mass map from noiseless shear field
      caused by NFW halo.
