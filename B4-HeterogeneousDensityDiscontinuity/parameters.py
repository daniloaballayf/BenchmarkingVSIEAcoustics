'''
Module contains simulation related parameters.
'''
import numpy as np

SIZE_EXT = 7
SIZE_INT = 5
WAVESPEED = 1482300                   # Exterior wavespeed
WAVESPEED_1 = WAVESPEED/1.2           # Mid wavespeed
WAVESPEED_2 = WAVESPEED/1.2           # Interior wavespeed
LAMBDA_EXT = 2.5                      # Exterior wavelength
FREC = WAVESPEED/LAMBDA_EXT           # Exterior frecuency
RHO_0 = 1000                          # Exterior density
RHO_1 = 1000                          # Mid density
RHO_2 = 2000                          # Interior density
LAMBDA_ERR = 3.5 
FREC_ERR = WAVESPEED/LAMBDA_ERR

INTDENSITYFUN = lambda x, y, z: RHO_2 - 800*np.sin(2*np.pi*x/SIZE_INT)*np.sin(2*np.pi*y/SIZE_INT)*np.cos(np.pi*z/SIZE_INT)
EXTDENSITYFUN = lambda x, y, z: RHO_1
INTWAVESPEEDFUN = lambda x, y, z: WAVESPEED_2
EXTWAVESPEEDFUN = lambda x, y, z: WAVESPEED_1

DXINTDENSITYFUN = lambda x, y, z: -800*(2*np.pi/SIZE_INT)*np.cos(2*np.pi*x/SIZE_INT)*np.sin(2*np.pi*y/SIZE_INT)*np.cos(np.pi*z/SIZE_INT)
DYINTDENSITYFUN = lambda x, y, z: -800*(2*np.pi/SIZE_INT)*np.sin(2*np.pi*x/SIZE_INT)*np.cos(2*np.pi*y/SIZE_INT)*np.cos(np.pi*z/SIZE_INT)
DZINTDENSITYFUN = lambda x, y, z: 800*(np.pi/SIZE_INT)*np.sin(2*np.pi*x/SIZE_INT)*np.sin(2*np.pi*y/SIZE_INT)*np.sin(np.pi*z/SIZE_INT)

DXEXTDENSITYFUN = lambda x, y, z: 0
DYEXTDENSITYFUN = lambda x, y, z: 0
DZEXTDENSITYFUN = lambda x, y, z: 0

PARAMETERS = {
    "SIZE_EXT": SIZE_EXT,
    "SIZE_INT": SIZE_INT,
    "WAVESPEED": WAVESPEED,
    "WAVESPEED_1": WAVESPEED_1,
    "WAVESPEED_2": WAVESPEED_2,
    "LAMBDA_EXT": LAMBDA_EXT,
    "FREC": FREC,
    "RHO_0": RHO_0,
    "RHO_1": RHO_1,
    "RHO_2": RHO_2,
    "INTDENSITYFUN": INTDENSITYFUN,
    "EXTDENSITYFUN": EXTDENSITYFUN,
    "INTWAVESPEEDFUN": INTWAVESPEEDFUN,
    "EXTWAVESPEEDFUN": EXTWAVESPEEDFUN,
    "DXINTDENSITYFUN": DXINTDENSITYFUN,
    "DYINTDENSITYFUN": DYINTDENSITYFUN,
    "DZINTDENSITYFUN": DZINTDENSITYFUN,
    "DXEXTDENSITYFUN": DXEXTDENSITYFUN,
    "DYEXTDENSITYFUN": DYEXTDENSITYFUN,
    "DZEXTDENSITYFUN": DZEXTDENSITYFUN,
    "LAMBDA_ERR": LAMBDA_ERR,
    "FREC_ERR": FREC_ERR
}