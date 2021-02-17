# SBFit

SBFit is an X-ray source surface brightness fitting tool.

## Installation

```shell
$ python setup.py install
```

## Simple tutorial

### Prerequisites

#### Image products:

Each observation / instrument needs three images:

- Count map
- Exposure map
- NXB map

Images of all observations / instruments need to have the same size.

#### DS9 format Region file:

The region file need to contain **one** inclusive region and any number of
exclusive regions.

Inclusive regions:

- Panda
- Epanda
- Projection

Exclusive regions:

- circle
- ellipse

## Working environment

Users are recommended to conduct data analysis in an interactive environment,
e.g. Jupyter Notebook.

## Tips

### Exposure map

The exposure map should include vignetting correction information and can have
different kinds of unit e.g. "s", "s*cm^2".

### Background map

sbfit supports two types of background scaling:

- count
- count rate

In either way, the scaling factor need to be provided in the NXB fits header.
The default keyword is `BKGNORM`. When use count rate scaling, an `EXPOSURE`
keyword is needed to calculate the NXB flux.

### Surface brightness calculation
In each pixel:
- net_count_rate = (count - scaled_bkg) / exp
- surface_brightness = net_count_rate / pixel_area

