In the cluster detection region, we focus on high peaks (SNR $\geq 4$).


# Science Goal

## "Fair Sample" Hypotheses:

Determine matter density parameter $\Omega_M$ by comparing cluster baryon
fraction to total cluster mass measurements.

## Truly Dark Clusters:

Dark viralized mass concentrations. We are not interested non-viralized dark
matter or projections.

If confirmed: Challenge galaxy formation paradigm, feedback supressing the
galaxy formation process?

## Mass-Observables Relation:

observables:

- central galaxy flux
- richness
- velocity dispersion
- X-ray temperature
- SZ decrement

# $2$-D
+ [Schneider (1996)](https://ui.adsabs.harvard.edu/abs/1996MNRAS.283..837S/abstract)

The paper proposes to detect clusters from the $2$-D mass map using aperture mass map reconstructions.

The aperture mass maps optimize the detection of structures with isothermal profile,
with a detection limit of velocity dispersion $\sim 600kms~h^{-1}$

+ [Hamana (2004)](https://ui.adsabs.harvard.edu/abs/2004MNRAS.350..893H/abstract)

Predict the high peak counts (SNR>4) using NFW halo profile and P-S halo mass function.

Study the detection rate and false detections.

## Bias and Scatter

### Detection bias (orientation):
+ [Hamana (2012)](https://ui.adsabs.harvard.edu/abs/2012MNRAS.425.2287H/abstract)


The correlation between peak height and masses is better for
$M_{1000}$ than virial mass.

The halo shape scatters the peak height.

A selection bias on halo's orientation: select cluster with
major axes aligned with the line-of-sight direction.


### Optimization:

+ [Laura (2012)](https://ui.adsabs.harvard.edu/abs/2012MNRAS.423.1711M/abstract)


### Mass Bias:

+ [Chen (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...891..139C/abstract)


## Observational work:

+ [Wittman (2001)](https://ui.adsabs.harvard.edu/abs/2001ApJ...557L..89W/abstract)

## CFHT

+ [Gavazzi (2007)](https://ui.adsabs.harvard.edu/abs/2007A%26A...462..459G/abstract)
    + $4$ square degree CFHTLS;
    + $\nu>3.5$;
    + redshift $0.1\sim 0.7$.

+ [Shan (2012)](https://ui.adsabs.harvard.edu/abs/2012ApJ...748...56S/abstract)
    + $64$ square degree CFHTLS;
    + $\nu>3.5$;
    + 126/301 peaks.

## Subaru

+ [Miyazaki (2018)](https://ui.adsabs.harvard.edu/abs/2018PASJ...70S..27M/abstract)
    + HSC S16A;
    + 160 square degree;
    + $\nu>4.7$;
    + 65 peaks.
+ [Hamana (2020)](https://ui.adsabs.harvard.edu/abs/2020PASJ..tmp..224H/abstract)
    + Mitigate dilution (with different redshift cuts);
    + HSC 120 square degree;
    + 107/124 peaks.

# $3$-D

## Detection only

+ [Hennawi (2005)](https://ui.adsabs.harvard.edu/abs/2005ApJ...624...59H/abstract)

The method finds the $3$-D kernel which is maximally related to
the tomographic shear measurements and identify the $3$-D position
of the kernel as the center of a cluster.

Study the accuracy and precision of clusters' redshifts measured
from photometric redshift surveys using particle-mesh (PM)
$N$-body simulations.

Tomography Matched Filtering (TMF) is introduced to optimally weight the source
galaxies. TMF enhances the detection by $76\%$ for $S/N \gt 4.5$.  TMF combines
tomographic and matched filtering (similar to matched filtering algorithms
used to find clusters in optical surveys)

Lasso is a extension of matched filter technique. The first iteration of lasso
is the matched filter detection.

In Hennawi (2005), the filter profile in the transverse plane is
$$\gamma(\vec{\theta};\vec{\theta}_0)=-\mathcal{R}[K(\frac{\vec{|\theta}-\vec{\theta}_0|}{\theta_s})
e^{-2i\phi}]$$,

## Mass Maps
+ [Glimpse3D Leonard 2014](https://ui.adsabs.harvard.edu/abs/2014MNRAS.440.1281L/abstract)

+ [SVD (VanderPlas 2011)](https://ui.adsabs.harvard.edu/abs/2011ApJ...727..118V/abstract)


