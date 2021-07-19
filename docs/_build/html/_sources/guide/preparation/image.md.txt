# Images

The `sbfit` package asks the users to provide three types of FITS images, i.e.,
count map, exposure map with vignetting correction, and instrumental background
map.

## Image types

### Count map

A count map is a raw image directly extracted from an event file. It includes
counts from:

- astronomical X-ray photons through the mirror
- X-ray photons from the instrumental fluorescent lines
- events that induced by cosmic rays

The second and third components are usually measured using observations with
unexposed detectors, e.g., the filter wheel closed observations of *XMM-Newton*
EPIC and the stowed observations of *Chandra* ACIS.

### Exposure map

An exposure map is used to correct the vignetting effect of the mirror. For
different telescopes, the default unit of an exposure map is different. It
can be `s` or `s cm^2`. The latter one is recommended for cross-instrument
comparison.

### Instrumental background map

An instrumental background map, aka non-X-ray background (NXB) map, is used to
evaluate the NXB level. The `sbfit` package supports two types of background
scaling:
- count
- count rate

In either way, the scaling factor need to be provided in the NXB fits header.
The default keyword is `BKGNORM`. When using count rate scaling, an `EXPOSURE`
keyword is required to calculate the NXB flux.

## Multiple instruments/observations

Many X-ray observatories have more than one telescope-detector module,
e.g., three for *XMM-Newton*, four for *Suzaku*, seven for *SRG*-eROSITA.
The `sbfit` package supports loading image sets from different detectors.
Before loading them, the images of different instruments need to be
reprojected to one frame. The reprojected images are all of the same size and
the same aiming point.

Although *Chandra* has only one telescope module, many objects have multiple
observation runs. Therefore, to analyze an object with multiple *Chandra*
observations, the images need to be reprojected as well.

## Energy range

The energy range for count map extraction depends on many factors. For galaxy
clusters, the temperature is about several keV. The cooling function in a
soft band, e.g., 0.5 -- 2.0 keV, is roughly independent of the temperature. 
