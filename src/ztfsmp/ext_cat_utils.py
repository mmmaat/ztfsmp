#!/usr/bin/env python3

from astropy.time import Time


gaia_edr3_refmjd = Time(2016.0, format='byear').mjd


mag2extcatmag = {'gaia': {'zg': 'BPmag',
                          'zr': 'RPmag',
                          'zi': 'RPmag'}, # Not a great workaround
                 'ps1': {'zg': 'gmag',
                         'zr': 'rmag',
                         'zi': 'imag'},
                 'ubercal_fluxcatalog': {'zg': 'zgmag',
                                         'zr': 'zrmag',
                                         'zi': 'zimag'},
                 'ubercal_fluxcatalog_or': {'zg': 'zgmag',
                                         'zr': 'zrmag',
                                         'zi': 'zimag'},
                 'ubercal_self': {'zg': 'zgmag',
                                  'zr': 'zrmag',
                                  'zi': 'zimag'},
                 'ubercal_ps1': {'zg': 'zgmag',
                                 'zr': 'zrmag',
                                 'zi': 'zimag'},
                 'ubercal_repop': {'zg': 'zgmag',
                                   'zr': 'zrmag',
                                   'zi': 'zimag'}}


emag2extcatemag = {'gaia': {'zg': 'e_BPmag',
                            'zr': 'e_RPmag',
                            'zi': 'e_RPmag'},
                   'ps1': {'zg': 'e_gmag',
                           'zr': 'e_rmag',
                           'zi': 'e_imag'},
                   'ubercal_fluxcatalog': {'zg': 'ezgmag',
                                           'zr': 'ezrmag',
                                           'zi': 'ezimag'},
                   'ubercal_fluxcatalog_or': {'zg': 'ezgmag',
                                           'zr': 'ezrmag',
                                           'zi': 'ezimag'},
                 'ubercal_self': {'zg': 'ezgmag',
                                  'zr': 'ezrmag',
                                  'zi': 'ezimag'},
                 'ubercal_ps1': {'zg': 'ezgmag',
                                 'zr': 'ezrmag',
                                 'zi': 'ezimag'},
                 'ubercal_repop': {'zg': 'ezgmag',
                                   'zr': 'ezrmag',
                                   'zi': 'ezimag'}}


extcat2colorstr = {'gaia': "B_p-R_p",
                   'ps1': "m_g-m_i"}


def ps1_cat_remove_bad(df):
    flags_bad = {'icrf_quasar': 4,
                 'likely_qso': 8,
                 'possible_qso': 16,
                 'likely_rr_lyra': 32,
                 'possible_rr_lyra': 64,
                 'variable_chi2': 128,
                 'suspect_object':536870912,
                 'poor_quality': 1073741824}

    flags_good = {'quality_measurement': 33554432,
                  'quality_stack': 134217728}

    df = df.loc[~(df['f_objID'] & sum(flags_bad.values())>0)]
    return df.loc[(df['f_objID'] & sum(flags_good.values())>0)]


def get_ubercal_catalog_in_cone(name, ubercal_config_path, center_ra, center_dec, radius, filtercode=None):
    with open(ubercal_config_path, 'r') as f:
        ubercal_config = yaml.load(f, Loader=yaml.Loader)

    def _get_cat(filtercode):
        cat_pos_df = pd.read_parquet(pathlib.Path(ubercal_config['paths']['ubercal']).joinpath(ubercal_config['paths'][name][filtercode]), columns=['Source', 'ra', 'dec'], engine='pyarrow').set_index('Source')
        mask = filter_catalog_in_cone(cat_pos_df, center_ra, center_dec, radius)
        gaiaids = cat_pos_df.loc[mask].index.tolist()
        del cat_pos_df

        cat_df = pd.read_parquet(pathlib.Path(ubercal_config['paths']['ubercal']).joinpath(ubercal_config['paths'][name][filtercode]), filters=[('Source', 'in', gaiaids)], engine='pyarrow').set_index('Source')
        cat_df = cat_df.loc[cat_df['n_obs']>=ubercal_config['config']['min_measure']]

        if name == 'fluxcatalog' or name == 'fluxcatalog_or' or name == 'repop':
            cat_df = cat_df.loc[cat_df['calflux_weighted_mean']>0.]
            # cat_df['calflux_rms'] = cat_df['calflux_weighted_std']
            # cat_df['calflux_weighted_std'] = cat_df['calflux_weighted_std']/np.sqrt(cat_df['n_obs']-1)
            cat_df = cat_df.assign(calflux_rms=cat_df['calflux_weighted_std'],
                                   calflux_weighted_std=cat_df['calflux_weighted_std']/np.sqrt(cat_df['n_obs']-1))


            cat_df = cat_df.assign(calmag_weighted_mean=-2.5*np.log10(cat_df['calflux_weighted_mean'].to_numpy()),
                                   calmag_weighted_std=2.5/np.log(10)*cat_df['calflux_weighted_std']/cat_df['calflux_weighted_mean'],
                                   calmag_rms=2.5/np.log(10)*cat_df['calflux_rms']/cat_df['calflux_weighted_mean'])

            cat_df.drop(labels=['calflux_weighted_mean', 'calflux_weighted_std', 'calflux_rms'], axis='columns', inplace=True)

        return cat_df

    if filtercode is None:
        cat_g_df = _get_cat('zg')
        cat_r_df = _get_cat('zr')
        cat_i_df = _get_cat('zi')

        common_stars = list(set(set(cat_g_df.index.tolist()) & set(cat_r_df.index.tolist()) & set(cat_i_df.index.tolist())))

        cat_g_df = cat_g_df.filter(items=common_stars, axis=0)
        cat_r_df = cat_r_df.filter(items=common_stars, axis=0)
        cat_i_df = cat_i_df.filter(items=common_stars, axis=0)

        cat_g_df.rename(columns={'calmag_weighted_mean': 'zgmag', 'calmag_weighted_std': 'ezgmag', 'calmag_rms': 'zgrms', 'n_obs': 'zg_n_obs', 'chi2_Source_res': 'zg_chi2_Source_res'}, inplace=True)
        cat_df = pd.concat([cat_g_df, cat_r_df[['calmag_weighted_mean', 'calmag_weighted_std', 'calmag_rms', 'n_obs', 'chi2_Source_res']].rename(columns={'calmag_weighted_mean': 'zrmag', 'calmag_weighted_std': 'ezrmag', 'calmag_rms': 'zrrms', 'n_obs': 'zr_n_obs', 'chi2_Source_res': 'zr_chi2_Source_res'}),
                            cat_i_df[['calmag_weighted_mean', 'calmag_weighted_std', 'calmag_rms', 'n_obs', 'chi2_Source_res']].rename(columns={'calmag_weighted_mean': 'zimag', 'calmag_weighted_std': 'ezimag', 'calmag_rms': 'zirms', 'n_obs': 'zi_n_obs', 'chi2_Source_res': 'zi_chi2_Source_res'})], axis=1)

        return cat_df
    else:
        return _get_cat(filtercode)
