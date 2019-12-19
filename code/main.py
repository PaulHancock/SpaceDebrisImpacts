#! /usr/bin/env python

from __future__ import print_function, division

import astropy
from astropy.table import Table
import numpy as np
import datetime
from skyfield.api import Topos, load, utc

__author__ = 'Paul Hancock'
__date__ = '2019-12-17'


# Compute the MWA location
MWA = Topos('26.70331940 S', '116.67081524 E')

def get_sats(kind='stations', key_type='str'):
    """
    Load a set of satellites

    parameters
    ----------
    kind : str
        The kind of satellites to be loaded. Available is:
        'stations'
        'starlink'
        other - geos

    key_type : str
        Default is to have numeric and string ids for each satellites (ie two ids each)
        If key_type=='str' then only use string ids, otherwise use only numeric ids

    returns
    -------
    sats : [:class:`skyfield.sgp4lib.EarthSatellite`,...]
    """
    if kind == 'stations':
        url = 'http://www.celestrak.com/NORAD/elements/stations.txt'
    elif kind == 'starlink':
        url = 'http://www.celestrak.com/NORAD/elements/starlink.txt'
    else:
        url = 'http://www.celestrak.com/NORAD/elements/geo.txt'
    satellites = load.tle(url)

    if key_type =='str':
        sat = [satellites[n] for n in satellites.keys() if isinstance(n,str)]
    else:
        sat = [satellites[n] for n in satellites.keys() if not isinstance(n,str)]
    return sat


def get_time_rage(start=None, duration=3600, step_size=1):
    """
    """
    if start is None:
        start = datetime.datetime.utcnow()
    steps = np.arange(0,duration,step_size)

    ts = load.timescale()
    trange = ts.utc(year=start.year,
                    month=start.month,
                    day=start.day,
                    hour=start.hour,
                    minute=start.minute,
                    second=steps)
    return trange


def get_broadcast_locations(freq_min=88e6, freq_max=108e6,
                            min_eirp=1e3):
    """
    """
    basedir = 'data/ACMA/'
    device_details = Table.read(f"{basedir}device_details.csv")
    site = Table.read(f"{basedir}site.csv")
    # There are >1million devices listed, select just those licenced in the FM band
    FM = device_details[np.where( (device_details['FREQUENCY'] < freq_max) & (device_details['FREQUENCY']>freq_min))]
    # Add the site information
    joined = astropy.table.join(FM, site, 'SITE_ID')
    # Select only transmitters with EIRP that is at least 1kW
    bright = joined[np.where(joined['EIRP']>min_eirp)]
    return bright


def get_rcs_dict(infile='https://www.celestrak.com/pub/satcat.txt'):
    """
    Load the rcs catalogue into a dictionary

    parameters
    ----------
    infile : string
        filename for the satcat file

    returns
    -------
    rcs : dict
        Dictionary of {name:rcs}
    """
    lines = open(infile.split('/')[-1]).readlines()
    rcs = {}
    for l in lines:
        nid = l[13:18]
        name = l[23:47].strip()
        area = l[119:127]
        try:
            rcs[f'{name}'] = float(area)
        except ValueError as e:
            pass
    return rcs


def alt_az_dist(grnd, satellite, times):
    """
    Compute the alt az and distance between ground and satellite at a given set of times

    parameters
    ----------
    grnd : :class:`skyfield.api.Topos`
        The ground station of interest

    satellite : :class:`skyfield.sgp4lib.EarthSatellite`
        The satellite(s) of interest

    times : :class:`skyfield.timelib.Time`
        Time(s) of interest

    returns
    -------
    alt, az, distance : float
        Units are degrees, degrees, km
    """
    difference = [s-grnd for s in satellite]
    alt, az, distance = zip(*[d.at(times).altaz() for d in difference])
    alt = np.array([a._degrees for a in alt])
    az = np.array([a._degrees for a in az])
    distance = np.array([a.km for a in distance])
    return alt, az, distance


def MWA_response(alt, az=0):
    """
    Basic MWA beam response which is just cos(ZA)

    parameters
    ----------
    alt, az : float
        Pointing direction in degrees

    returns
    -------
    gain : float
        Gain of the telescope, normalised to 1
    """
    # TODO: look up values based on the MWA primary beam (empirical or full_EE)
    ZA = 90 - np.clip(alt,0, 90)
    return np.cos(np.radians(ZA))**2


def transmitter_response(alt, n=6, spacing=3, tilt=0):
    """
    Short cut all the Jones matrix calcs and give a direct solution

    parameters
    ----------
    alt : float
        Altitude pointing

    n : int
        Number of bays for compound antenna

    spacing : float
        The spacing (in lambda) between adjacent antennae

    tilt : float
        Introduce a phase offset such that the peak emission is `tilt` degrees below the horizon.

    returns
    -------
    gain : float
        Gain of the telescope, normalised to 1 at max
    """
    theta = 90-np.array(alt)
    omega = np.zeros_like(theta, dtype=np.complex)
    for i in range(n):
        dphi = i*spacing*np.cos(np.radians(theta))  + np.radians(tilt)
        omega += np.exp(1j*dphi)
    omega /=n
    E_theta = np.sin(np.radians(theta))
    return np.abs(omega * E_theta)**2


def power(site, t_distance, r_distance, t_elevation, r_elevation, RCS=1):
    """
    See https://ieeexplore.ieee.org/document/7971960
    Received peak power is:

           Pt * Gr * Gt * RCS
    Pr = ----------------------    [W / m^2]
         (4*pi)^2 * Rt^2 * Rr^2


    S = Pr/1e-26 / bandwidth(Hz) [Jy]

    parameters
    ----------
    site : {'EIRP':float, 'BANDWIDTH':float}
        Transmitter information

    t_distance, r_distance : float
        distance in km from transmitter to target, and target to receiver

    t_elevation, r_elevation: float
        elevation in degrees of the target as seen by the transmitter and receiver

    RCS : float
        radar cross section in m^2, default=1m^2

    returns
    -------
    Pr, Jy : float, float
        recieved power in W/m^2 and in Jy
    """
    Tx = transmitter_response(t_elevation)
    Rx = MWA_response(r_elevation)
    Pt = site['EIRP']
    Pr = Pt * Rx * Tx * RCS
    Pr /= (4*np.pi)**2 * (t_distance*1e3)**2 * (r_distance*1e3)**2
    Jy = Pr / (1e-26 * site['BANDWIDTH']/10e3) # 10e3 is the receiver BW
    return Pr, Jy


if __name__ == "__main__":
    print("build satellites")
    satellites = get_sats(kind='starlink')
    print(f'found {len(satellites)} satellites')
    with open('sat_names.txt','w') as out:
        for s in satellites:
            print(f'{s.name}', file=out)

    print("set up time step")
    now = datetime.datetime(2019, 11, 6, 12, 37, 37, 0)
    now = now.replace(tzinfo=utc)
    t = get_time_rage(start=now, duration=3600, step_size=10)
    with open('times.txt', 'w') as out:
        for time in t:
            print(time.utc_datetime(), file=out)

    print("Pre-compute satellite locations from MWA")
    MWA_alt, MWA_az, MWA_dist = alt_az_dist(MWA, satellites, t)
    np.save('MWA_alt.npy',MWA_alt)
    np.save('MWA_az.npy', MWA_az)
    np.save('MWA_dist.npy', MWA_dist)

    print("load ground stations")
    bc = get_broadcast_locations()
    bc = bc[bc['STATE']=='WA']
    print("convert to Topos")
    transmitters = [Topos(longitude_degrees=i, latitude_degrees=j) for (i,j) in bc['LONGITUDE','LATITUDE']]
    print(f'created {len(transmitters)} transmitters')
    with open('tx_names.txt','w') as out:
        for tx in bc:
            print(f'{tx["NAME"]}', file=out)
    freqs = np.array(bc['FREQUENCY'])
    np.save('tx_freqs.npy', freqs)

    print("Compute powers") 
    P = np.zeros(shape=(len(transmitters), len(satellites), len(t)))
    for i, gnd in enumerate(transmitters):
        print(f'station [{i}/{len(transmitters)}] {bc[i]["NAME"]}')
        alt, az, dist = alt_az_dist(gnd, satellites, t)
        # alt.shape = (len(satellites), len(t))
        P[i,:,:] = power(bc[i], dist, MWA_dist, alt, MWA_alt)[1]
        # power[0].shape == alt.shape == MWA_alt.shape == (len(satellites, len(t)))
    np.save('power.npy', P)

