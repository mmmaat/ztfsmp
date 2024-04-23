#!/usr/bin/env python3

import numpy as np


filtercodes = ['zg', 'zr', 'zi']
quadrant_width_px, quadrant_height_px = 3072, 3080
quadrant_size_px = {'x': quadrant_width_px, 'y': quadrant_height_px}


ztf_longitude = -116.8598 # deg
ztf_latitude = 33.3573 # deg E
ztf_altitude = 1668. # m


filtercode2ztffid = {'zg': 1,
                     'zr': 2,
                     'zi': 3}


filtercode2color = {'zg': 'green',
                    'zr': 'red',
                    'zi': 'orange'}


def plot_ztf_focal_plan(fig, focal_plane_dict, plot_fun, plot_ccdid=False):
    ccds = fig.add_gridspec(4, 4, wspace=0.02, hspace=0.02)
    for i in range(4):
        for j in range(4):
            ccdid = 16 - (i*4+j)
            quadrants = ccds[i, j].subgridspec(2, 2, wspace=0., hspace=0.)
            axs = quadrants.subplots()

            for k in range(2):
                for l in range(2):
                    rcid = (ccdid-1)*4 + k*2
                    qid = k*2
                    if k > 0:
                        rcid += l
                        qid += l
                    else:
                        rcid -= (l - 1)
                        qid -= (l - 1)

                    plot_fun(axs[k, l], focal_plane_dict[ccdid][qid], ccdid, qid, rcid)

            if plot_ccdid:
                ax = fig.add_subplot(ccds[i, j])
                ax.text(0.5, 0.5, ccdid, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontweight='black', fontsize='xx-large')
                ax.axis('off')

    for ax in fig.get_axes():
        ss = ax.get_subplotspec()
        ax.spines.top.set_visible(ss.is_first_row())
        ax.spines.top.set(linewidth=1.)
        ax.spines.bottom.set_visible(ss.is_last_row())
        ax.spines.bottom.set(linewidth=1.)
        ax.spines.left.set_visible(ss.is_first_col())
        ax.spines.left.set(linewidth=1.)
        ax.spines.right.set_visible(ss.is_last_col())
        ax.spines.right.set(linewidth=1.)



def plot_ztf_focal_plan_rcid(fig):
    rcids = dict([(i+1, dict([(j, j) for j in range(4)])) for i in range(0, 16)])

    def _plot(ax, val, ccdid, qid, rcid):
        ax.text(0.5, 0.5, rcid, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    plot_ztf_focal_plan(fig, rcids, _plot, plot_ccdid=True)


def plot_ztf_focal_plan_values(fig, focal_plane_dict, scalar=False, vmin=None, vmax=None, cmap=None):
    def _plot(ax, val, ccdid, qid, rcid):
        if val is not None:
            if scalar:
                val = [[val]]

            ax.imshow(val, vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set(xticks=[], yticks=[])
            ax.set_aspect('auto')

        ax.text(0.5, 0.5, rcid+1, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    if vmin is None or vmax is None:
        values = list(filter(lambda x: x is not None, values))

        if vmin is None:
            vmin = np.min(values)

        if vmax is None:
            vmax = np.max(values)

    plot_ztf_focal_plan(fig, focal_plane_dict, _plot, plot_ccdid=True)


def ztf_quadrant_name_explode(quadrant_name, kind='sci'):
    quadrant_name = "_".join(quadrant_name.split("_")[1:])

    if kind == 'sci':
        year = int(quadrant_name[0:4])
        month = int(quadrant_name[4:6])
        day = int(quadrant_name[6:8])
        field = int(quadrant_name[15:21])
        filterid = quadrant_name[22:24]
        ccdid = int(quadrant_name[26:28])
        qid = int(quadrant_name[32])

        return year, month, day, field, filterid, ccdid, qid

    elif kind == 'raw':
        assert False, "ztf_quadrant_name_explode(): needs to be updates/tested for raw image name"
        year = int(quadrant_name[0:4])
        month = int(quadrant_name[4:6])
        day = int(quadrant_name[6:8])
        field = int(quadrant_name[15:21])
        filterid = quadrant_name[22:24]
        ccdid = int(quadrant_name[26:28])

        return year, month, day, field, filterid, ccdid

    else:
        raise NotImplementedError("Kind {} not implemented!".format(kind))


def ztfquadrant_center(wcs):
    return wcs.pixel_to_world([quadrant_width_px/2.], [quadrant_height_px/2.])
