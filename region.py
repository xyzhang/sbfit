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
    # TODO Now only support ds9 format
    region_types = {"circle": Circle, "panda": Panda, "epanda": Epanda}
    region_list = RegionList()
    if style == "ds9":
        frame = purified[1]
        for line in purified[2:]:
            if line[0] == "-":
                status: str = "SUB"
                rtype = line[1:].split("(")[0]
            else:
                status = "ADD"
                rtype = line.split("(")[0]
            parameters = line.split("(")[1].split(")")[0].split(",")
            if rtype not in ["circle", "epanda", "panda"]:
                print(f"Unsupported region type: {rtype}")
            else:
                if status == "ADD":
                    region_list.add = region_types[rtype](frame, *parameters, status)
                elif status == "SUB":
                    region_list.sub += [region_types[rtype](frame, *parameters, status)]
    else:
        raise TypeError("So far, we only support ds9 format.")
    return region_list
    # TODO Now only support one region per file.


class Region(object):

    def __init__(self):
        self.status = "ADD"


class Circle(Region):

    def __init__(self, frame="image", x=0, y=0, radius=1, status="ADD"):
        super().__init__()
        self.frame = frame
        self.status = status
        if frame == "image":
            # Now the unit of coordinates is pixel.
            self.x = float(x)
            self.y = float(y)
            self.radius = float(radius)
        elif frame in ["fk4", "fk5", "icrs", "galactic", "ecliptic"]:
            center: coordinates.SkyCoord = coordinates.SkyCoord(x, y, frame=frame, unit=(units.hourangle, units.deg))
            self.x = center.ra
            self.y = center.dec
            if radius[-1] == '"':
                self.radius = float(radius[:-1]) * units.arcsec
            else:
                raise TypeError("Sky coordinate unit must be arcsec.")


class Epanda(Region):

    def __init__(self, frame="image", x=0, y=0, startangle=0, stopangle=360, nangle=1, innermajor=0, innerminor=0,
                 outermajor=100, outerminor=100, nradius=1, angle=0, status="ADD"):
        super().__init__()
        self.frame = frame
        self.status = status
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


class Panda(Region):

    def __init__(self, frame="image", x=0, y=0, startangle=0, stopangle=360, nangle=1, inner=0,
                 outer=100, nradius=1, status="ADD"):
        super().__init__()
        self.frame = frame
        self.status = status
        self.startangle = float(startangle)
        self.stopangle = float(stopangle)
        self.nangle = int(nangle)
        self.nradius = int(nradius)
        self.angle = 0.0
        if frame == "image":
            # Now the unit of coordinates is pixel.
            self.x = float(x)
            self.y = float(y)
            self.innermajor = float(inner)
            self.innerminor = float(inner)
            self.outermajor = float(outer)
            self.outerminor = float(outer)
        elif frame in ["fk4", "fk5", "icrs", "galactic", "ecliptic"]:
            center: coordinates.SkyCoord = coordinates.SkyCoord(x, y, frame=frame, unit=(units.hourangle, units.deg))
            self.x = center.ra
            self.y = center.dec
            if inner[-1] == '"':
                self.innermajor = float(inner[:-1]) * units.arcsec
                self.innerminor = float(inner[:-1]) * units.arcsec
                self.outermajor = float(outer[:-1]) * units.arcsec
                self.outerminor = float(outer[:-1]) * units.arcsec
            else:
                raise TypeError("Sky coordinate unit must be arcsec.")


class RegionList(object):

    def __init__(self):
        self.add = Region()
        self.sub = []
