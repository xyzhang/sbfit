import numpy as np
from astropy import units, coordinates


# TODO Test of this module


def read_region(filename, style="ds9"):
    """Load region parameters from a plan-text file."""
    global parameters
    with open(filename) as f:
        contents = f.read().split("\n")
    # Purify comments.
    purified = []
    for line in contents:
        if len(line) == 0:
            pass
        elif line.strip()[0] == "#":
            pass
        else:
            purified.append(line.strip().split("#")[0].strip())
    if style == "ds9":
        for line in purified:
            if "global" in line:
                pass
            elif line in ["image", "physical", "amplifier", "detector",
                          "fk4", "fk5", "icrs", "galactic", "ecliptic"]:
                frame = line
            elif "epanda" in line:
                if line[0] == "-":
                    status: str = "-"
                else:
                    status = "+"
                parameters = line.split("(")[1].split(")")[0].split(",")
            else:
                print(f"Unsupported region type: {line}")
    return Epanda(frame, *parameters, status)
    # TODO Now only support one region per file.


class Region(object):

    def __init__(self):
        self.status = "ADD"


class Epanda(Region):

    def __init__(self, frame="image", x=0, y=0, startangle=0, stopangle=360, nangle=1, innermajor=0, innerminor=0,
                 outermajor=100, outerminor=100, nradius=1, angle=0, status="+"):
        super().__init__()
        self.frame = frame
        if status == "+":
            self.status = "ADD"
        elif status == "-":
            self.status = "SUB"
        self.startangle = float(startangle)
        self.stopangle = float(stopangle)
        self.nangle = int(nangle)
        self.nradius = int(nradius)
        self.angle = float(angle)
        if frame == "image":
            # Now the unit of coordinates is pixel.
            self.x = float(x)
            self.y = float(y)
            self.innermajor = float(innermajor)
            self.innerminor = float(innerminor)
            self.outermajor = float(outermajor)
            self.outerminor = float(outerminor)
        elif frame in ["fk4", "fk5", "icrs", "galactic", "ecliptic"]:
            center: coordinates.SkyCoord = coordinates.SkyCoord(x, y, frame=frame, unit=(units.hourangle, units.deg))
            self.x = center.ra
            self.y = center.dec
            if innermajor[-1] == '"':
                self.innermajor = float(innermajor[:-1]) * units.arcsec
                self.innerminor = float(innerminor[:-1]) * units.arcsec
                self.outermajor = float(outermajor[:-1]) * units.arcsec
                self.outerminor = float(outerminor[:-1]) * units.arcsec
            else:
                raise TypeError
