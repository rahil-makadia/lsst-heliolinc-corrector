### Python Differential Corrector Helper Functions
"""
dc_helper_funcs.py

This file contains the functions necessary to run the Python Differential Corrector for LSST HelioLinc3D results
Implementation: Python 3.8, R. Makadia 09092022
"""

import numpy as np
from .integrator import spkFile, mjd2et, propagate_gr15
from spiceypy import furnsh, spkez
from astropy.time import Time
from skyfield.api import load, S, W, wgs84

for f in spkFile:
    furnsh(f)

zeros = np.zeros
empty = np.empty
norm = np.linalg.norm
inv = np.linalg.inv
sin = np.sin
cos = np.cos
asin = np.arcsin
atan2 = np.arctan2
pi = np.pi
array = np.array
diag = np.diag
ascontiguousarray = np.ascontiguousarray
hstack = np.hstack

au2km = 1.495978707e8
day2sec = 8.64e4

def get_equat_from_eclip(eclip_state):
    """Convert an ecliptic state to equatorial state

    Args:
        eclip_state (vector): 6-element vector consisting of ecliptic cartesian state

    Returns:
        equat_state (vector): 6-element vector consisting of equatorial cartesian state
    """
    eme_obliq = 84381.448/3600*np.pi/180 # EMEJ2000 obliquity, https://ssd.jpl.nasa.gov/sbdb.cgi?sstr=didymos;old=0;orb=0;cov=1;log=0;cad=1
    eclip2equat = array(([1,            0,                   0     ],
                         [0,    cos(eme_obliq),     -sin(eme_obliq)],
                         [0,    sin(eme_obliq),      cos(eme_obliq)])) # https://archive.org/details/131123ExplanatorySupplementAstronomicalAlmanac/page/n291/mode/2up\n,

    pos_equat = ascontiguousarray(eclip2equat) @ ascontiguousarray(eclip_state[:3])
    vel_equat = ascontiguousarray(eclip2equat) @ ascontiguousarray(eclip_state[3:6])
    
    return hstack((pos_equat, vel_equat))
# end def

def get_radec(state):
    """Convert a cartesian state into right ascension and declination

    Args:
        state (vector): 6-element cartesian state vector

    Returns:
        ra (float): right ascension in radians
        dec (float): declination in radians
    """
    pos = state[:3]
    vel = state[3:6]

    dist = norm(pos) # calculate distance
    ra = atan2(pos[1], pos[0])
    if ra < 0:
        ra = ra + 2*pi
    # end if
    dec = asin(pos[2]/dist) # calculate declination

    return ra, dec
# end def

def get_perturbed_state(x_nom, idx, fd_pert):
    """Get the perturbed nominal state for central differencing within differential corrector

    Args:
        x_nom (vector): nominal state vector for current differential corrector iteration
        idx (int): index for nominal state element to be perturbed
        fd_pert (float): factor to perturb nominal state element by

    Returns:
        x_plus (vector): pertubed state vector using the positive perturbation value
        x_minus (vector): pertubed state vector using the negative perturbation value
        fd_delta (float): perturbation value
    """
    x_plus = x_nom.copy()
    x_minus = x_nom.copy()
    fd_delta = x_nom[idx]*fd_pert # fd_pert = finite difference perturbation to nominal state for calculating derivatives

    x_plus[idx] = x_nom[idx]+fd_delta
    x_minus[idx] = x_nom[idx]-fd_delta

    return x_plus, x_minus, fd_delta
# end def

def accumulate_observations_efficiently(epoch, x_nom, obs_array, fd_pert=0.01):
    # sourcery skip: low-code-quality

    """Accumulate observation data and run one iteration of differential corrector

    Args:
        epoch (float): Julian date for which to return state estimate
        x_nom (vector): Initial guess for nominal state at epoch
        obs_array (array): Array of observation data to be used for orbit fit and beta estimate if DART=True
        fd_pert (float, optional): factor to perturb nominal state element by. Defaults to 0.01.

    Returns:
        P (array): Covariance matrix corresponding to the orbit fit at the end of iteration
        at_w_b (vector): weighted residual vector for computing the nominal state guess at next iteration
        b_accum (vector): reidual value for each observation for computing the RMS at each iteration
    """
    x_size = len(x_nom)
    num_obs = len(obs_array)
    a = zeros((2*num_obs, x_size))
    # at_w_a = zeros((x_size, x_size))
    # at_w_b = zeros((x_size, 1))
    b = empty((2*num_obs,1))
    # b_accum = empty((num_obs, 1))
    w = zeros((2*num_obs, 2*num_obs))

    # propagate nominal state to each observation epoch
    past_obs_exist = False
    present_future_obs_exist = False
    past_epoch_idx = np.where(obs_array[:,0]<epoch)[0]
    present_future_epoch_idx = np.where(obs_array[:,0]>=epoch)[0]
    if past_epoch_idx.size != 0:
        before_limit = np.min(obs_array[past_epoch_idx,0])
        past_obs_exist = True
    if present_future_epoch_idx.size != 0:
        after_limit = np.max(obs_array[present_future_epoch_idx,0])
        present_future_obs_exist = True
    if past_obs_exist:
        _, _, state_prop_to_obs_bef_epoch = propagate_gr15(epoch, x_nom, before_limit, t_eval=obs_array[past_epoch_idx,0])
    if present_future_obs_exist:
        _, _, state_prop_to_obs_aft_epoch = propagate_gr15(epoch, x_nom, after_limit, t_eval=obs_array[present_future_epoch_idx,0])

    if past_obs_exist and not present_future_obs_exist:
        state_prop_to_obs_arr = state_prop_to_obs_bef_epoch
    if not past_obs_exist and present_future_obs_exist:
        state_prop_to_obs_arr = state_prop_to_obs_aft_epoch
    if past_obs_exist and present_future_obs_exist:
        state_prop_to_obs_arr = np.vstack((state_prop_to_obs_bef_epoch, state_prop_to_obs_aft_epoch))

    # perturb each element of the nominal state in the positive and negative direction, propagate the perturbed state to each observation epoch, for later calculation of finite (central) differences
    state_prop_to_obs_plus_arr = zeros((x_size, num_obs, 6))
    state_prop_to_obs_minus_arr = zeros((x_size, num_obs, 6))

    for x_idx in range(x_size):
        x_plus, x_minus, fd_delta = get_perturbed_state(x_nom, x_idx, fd_pert)
        if past_obs_exist:
            # perturb in + direction before epoch
            _, _, state_prop_to_obs_plus_bef_epoch = propagate_gr15(epoch, x_plus, before_limit, t_eval=obs_array[past_epoch_idx,0])
            # perturb in - direction before epoch
            _, _, state_prop_to_obs_minus_bef_epoch = propagate_gr15(epoch, x_minus, before_limit, t_eval=obs_array[past_epoch_idx,0])
        if present_future_obs_exist:
            # perturb in + direction after epoch
            _, _, state_prop_to_obs_plus_aft_epoch = propagate_gr15(epoch, x_plus, after_limit, t_eval=obs_array[present_future_epoch_idx,0])
            # perturb in - direction after epoch
            _, _, state_prop_to_obs_minus_aft_epoch = propagate_gr15(epoch, x_minus, after_limit, t_eval=obs_array[present_future_epoch_idx,0])

        if past_obs_exist and not present_future_obs_exist:
            state_prop_to_obs_plus_arr[x_idx] = state_prop_to_obs_plus_bef_epoch
            state_prop_to_obs_minus_arr[x_idx] = state_prop_to_obs_minus_bef_epoch
        if not past_obs_exist and present_future_obs_exist:
            state_prop_to_obs_plus_arr[x_idx] = state_prop_to_obs_plus_aft_epoch
            state_prop_to_obs_minus_arr[x_idx] = state_prop_to_obs_minus_aft_epoch
        if past_obs_exist and present_future_obs_exist:
            state_prop_to_obs_plus_arr[x_idx] = np.vstack((state_prop_to_obs_plus_bef_epoch, state_prop_to_obs_plus_aft_epoch))
            state_prop_to_obs_minus_arr[x_idx] = np.vstack((state_prop_to_obs_minus_bef_epoch, state_prop_to_obs_minus_aft_epoch))

    lat = 30+14/60+40.68/3600 # lsst latitude, from https://www.lsst.org/scientists/keynumbers
    lon = 70+44/60+57.90/3600 # lsst longitude, from https://www.lsst.org/scientists/keynumbers
    elev = 2647 # lsst elevation, from https://www.lsst.org/scientists/keynumbers
    for obs_idx in range(num_obs):
        curr_obs = obs_array[obs_idx, :]
        time_obs = Time(curr_obs[0], format='mjd', scale='tdb')
        ra_obs = curr_obs[1]
        dec_obs = curr_obs[2]
        earth_state = spkez(399, mjd2et(time_obs.mjd), 'ECLIPJ2000', 'NONE', 0)[0]
        change_units = 1/au2km*np.ones_like(earth_state) # convert km to au
        change_units[3:6] = day2sec*change_units[3:6] # convert au/sec to au/day
        earth_state = earth_state*change_units

        ts = load.timescale()
        t = ts.from_astropy(time_obs)

        lsst = wgs84.latlon(lat*S, lon*W, elevation_m=elev)
        lsst_state_equat = np.hstack((lsst.at(t).position.au, lsst.at(t).velocity.au_per_d)) # equatorial lsst state

        state_prop_to_obs = state_prop_to_obs_arr[obs_idx, :]
        state_obs_equat = get_equat_from_eclip(state_prop_to_obs-earth_state) - lsst_state_equat
        ra_nom, dec_nom = get_radec(state_obs_equat)
        b[2*obs_idx+0, 0] = ra_obs - ra_nom
        b[2*obs_idx+1, 0] = dec_obs - dec_nom
        # b_accum[obs_idx] = b[2*obs_idx+0:2*obs_idx+2].T @ b[2*obs_idx+0:2*obs_idx+2]

        for x_idx in range(x_size):
            x_plus, x_minus, fd_delta = get_perturbed_state(x_nom, x_idx, fd_pert)
            # # perturb in + direction
            state_prop_to_obs_plus = state_prop_to_obs_plus_arr[x_idx][obs_idx, :]
            state_obs_equat_plus = get_equat_from_eclip(state_prop_to_obs_plus-earth_state)
            # perturb in - direction
            state_prop_to_obs_minus = state_prop_to_obs_minus_arr[x_idx][obs_idx, :]
            state_obs_equat_minus = get_equat_from_eclip(state_prop_to_obs_minus-earth_state)
            ra_plus, dec_plus = get_radec(state_obs_equat_plus)
            ra_minus, dec_minus = get_radec(state_obs_equat_minus)
            # populate partial derivative matrix for right ascension and declination using central differences
            a[2*obs_idx+0, x_idx] = (ra_plus - ra_minus)/(2*fd_delta)
            a[2*obs_idx+1, x_idx] = (dec_plus - dec_minus)/(2*fd_delta)
        # end for

        # populate observation weight matrix
        w[2*obs_idx+0, 2*obs_idx+0] = 1/curr_obs[3]**2
        w[2*obs_idx+1, 2*obs_idx+1] = 1/curr_obs[4]**2
    # end for

    # calculate differential correction covariance matrix
    P = inv(a.T @ w @ a)

    return P, a, w, b
# end def
