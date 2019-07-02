
# SBFit 
SBFit is an X-ray source surface brightness fitting tool.

## Installation
```shell
$ python setup.py install
```
## Tips
### Exposure map
The exposure map should include vignetting correction information 
and can have different kinds of unit e.g. "s", "s*cm^2".

### Background map
The background map is a particle background count map. 
So it should not be vignetting corrected. The unit
of background profile is always "cts/s/arcmin^2"

In this module, there are two background scaling methods. One is only using a scaling factor to scale the
background count number. Another one is using a scaling factor to scale the count rate. In the second
way, an EXPOSURE keyword is needed in the background file header.

### Flux calculation
flux = (count - scaled_bkg) / exp

