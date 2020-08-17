import numpy as np
from witt import witt

def logtau_to_height(logtau, temperature, ne):
    '''
    Convert optical depth scale to geometric height and column density (cgs).

    This function uses the Wittmann equation of state, ported to python by
    Jaime de la Cruz Rodriguez (ISP - SU)
    (https://github.com/jaimedelacruz/witt). This function based on
    atmosphere creation functions in Lightweaver, originally built on
    approach used in RH.

    Parameters
    ----------
    logtau : array_like
        Optical depth, stratified down the line of sight (i.e. for increasing values of logtau).

    temperature : array_like
        Temperature at each logtau point [K].

    ne : array_like
        Electron density at each logtau point [cm^-3].

    Returns
    -------

    height : array
        Altitude [cm].

    cmass : array
        Column density [g cm^-2].
    '''
    logtau = np.atleast_1d(logtau)
    temperature = np.atleast_1d(temperature)
    ne = np.atleast_1d(ne)
    # NOTE(cmo): Assumption is all units are cgs
    # Adbundances are assumed to be the RH defaults, this can be overriden in
    # the initialiser.
    eos = witt()

    # NOTE(cmo): Electron pressure from ne
    pe = ne * eos.BK * temperature
    # NOTE(cmo): gas pressure from EOS
    pgas = np.zeros_like(logtau)
    # NOTE(cmo): mass density from EOS
    rho = np.zeros_like(logtau)
    for k in range(logtau.shape[0]):
        pgas[k] = eos.pg_from_pe(temperature[k], pe[k])
        rho[k] = eos.rho_from_pe(temperature[k], pe[k])

    # NOTE(cmo): Continuum opacity (5000 \AA)
    chiC = np.zeros_like(logtau)
    for k in range(logtau.shape[0]):
        chiC[k] = eos.contOpacity(temperature[k], pgas[k], pe[k], np.array([5000.0]))

    # NOTE(cmo): Compute column mass and geometric height
    cmass = np.zeros_like(logtau)
    height = np.zeros_like(logtau)
    tauRef = 10**logtau
    cmass[0] = (tauRef[0] / chiC[0]) * rho[0]
    for k in range(1, logtau.shape[0]):
        height[k] = height[k-1] - 2.0 * (tauRef[k] - tauRef[k-1]) / (chiC[k-1] + chiC[k])
        cmass[k] = cmass[k-1] + 0.5 * (chiC[k-1] + chiC[k]) * (height[k-1] - height[k])

    # NOTE(cmo): Adjust height scale so height = 0 at log tau = 0
    tauUnityHeight = np.interp(1.0, tauRef, height)
    height -= tauUnityHeight

    return height, cmass
