#!/usr/bin/env python3

import numpy as np
import pandas as pd
from croaks.match import NearestNeighAssoc


idx2markerstyle = ['*', 'x', '.', 'v', '^']


def make_index_from_list(dp, index_list):
    """
    Calls make_index() of DataProxy dp on all index names in index_list
    """
    [dp.make_index(index) for index in index_list]


def sc_ra(skycoord):
    return skycoord.frame.data.lon.value


def sc_dec(skycoord):
    return skycoord.frame.data.lat.value


def sc_array(skycoord):
    return np.array([skycoord.frame.data.lon.value, skycoord.frame.data.lat.value])


def contained_in_exposure(objects, wcs, return_mask=False):
    width, height = wcs.pixel_shape

    if isinstance(objects, pd.DataFrame):
        mask = (objects['x'] >= 0.) & (objects['x'] < width) & (objects['y'] >= 0.) & (objects['y'] < height)
    else:
        top_left = [0., height]
        top_right = [width, height]
        bottom_left = [0., 0.]
        bottom_right = [width, 0]

        tl_radec = sc_array(wcs.pixel_to_world(*top_left))
        tr_radec = sc_array(wcs.pixel_to_world(*top_right))
        bl_radec = sc_array(wcs.pixel_to_world(*bottom_left))
        br_radec = sc_array(wcs.pixel_to_world(*bottom_right))

        tl = [max([tl_radec[0], bl_radec[0]]), min([tl_radec[1], tr_radec[1]])]
        br = [min([tr_radec[0], br_radec[0]]), max([bl_radec[1], br_radec[1]])]

        o = sc_array(objects)
        if tl[0] < br[0]:
            mask = (o[0] > tl[0]) & (o[0] < br[0]) & (o[1] > tl[1]) & (o[1] < br[1])
        else:
            mask = (o[0] < tl[0]) & (o[0] > br[0]) & (o[1] > tl[1]) & (o[1] < br[1])

    if return_mask:
        return mask
    else:
        return objects[mask]


def match_pixel_space(refcat, cat, radius=1.):
    def _euclidian(x1, y1, x2, y2):
        return np.sqrt((np.array(x1) - np.array(x2)) ** 2 + (np.array(y1) - np.array(y2)) ** 2)

    assoc = NearestNeighAssoc(first=[refcat['x'], refcat['y']], radius=radius)
    index = assoc.match(cat['x'], cat['y'], metric=_euclidian)
    return index


def write_ds9_reg_circles(output_path, positions, radius):
    with open(output_path, 'w') as f:
        f.write("J2000\n")
        for position, r in zip(positions, radius):
            f.write("circle {}d {}d {}p\n".format(position[0], position[1], r))


def create_2D_mesh_grid(*meshgrid_space):
    meshgrid = np.meshgrid(*meshgrid_space)
    return np.array([meshgrid[0], meshgrid[1]]).T.reshape(-1, 2)


def poly2d_from_file(filename):
    with open(filename, 'r') as f:
        f.readline()
        deg_str = f.readline()[:-1]
        degree = int(deg_str.split(" ")[1])
        coeff_str = " ".join(f.readline()[:-1].split())
        coeffs = list(map(float, coeff_str.split(" ")))

    coeffs_1 = coeffs[:int(len(coeffs)/2)]
    coeffs_2 = coeffs[int(len(coeffs)/2):]

    def _extract_coeffs(coeffs):
        idx = 0
        c = np.zeros([degree, degree])
        for d in range(degree):
            p, q = d, 0
            while p >= 0:
                c[p, q] = coeffs[idx]
                idx += 1
                p -= 1
                q += 1

        return c

    c_1 = _extract_coeffs(coeffs_1)
    c_2 = _extract_coeffs(coeffs_2)

    def _apply_pol(x, y):
        return np.stack([np.polynomial.polynomial.polyval2d(x, y, c_1),
                         np.polynomial.polynomial.polyval2d(x, y, c_2)])

    return _apply_pol


def poly2d_to_file(poly2d, quadrant_set, outdir):
    """
    dump the fitted transformations (in the poloka format)
    so that they can be understood by mklc
    """

    outdir.mkdir(exist_ok=True)

    for i, quadrant in enumerate(quadrant_set):
        #with open(outdir + os.sep + 'transfoTo' + expid + 'p' + ccd + '.dat', 'w') as f:
        with open(outdir.joinpath("transfoTo{}.dat".format(quadrant)), 'w') as f:
            deg = poly2d.bipol2d.deg
            f.write("GtransfoPoly 1\ndegree %d\n" % deg)

            coeff_name = dict(list(zip(poly2d.bipol2d.coeffs, [x for x in poly2d.bipol2d.coeffnames if 'alpha' in x])))
            for d in range(deg+1):
                p,q = d,0
                while p>=0:
                    nm = coeff_name[(p,q)]
                    scaled_par = poly2d.params[nm].full[i]
                    f.write(" %15.20E " % scaled_par)
                    p -= 1
                    q += 1
            coeff_name = dict(list(zip(poly2d.bipol2d.coeffs, [x for x in poly2d.bipol2d.coeffnames if 'beta' in x])))
            for d in range(deg+1):
                p,q = d,0
                while p>=0:
                    nm = coeff_name[(p,q)]
                    scaled_par = poly2d.params[nm].full[i]
                    f.write(" %15.20E " % scaled_par)
                    p -= 1
                    q += 1
            f.write('\n')
