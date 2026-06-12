'''
Module contains simulation related parameters.
'''
import numpy as np

SIZE = 7e-3
WAVESPEED = 1482.3                    # Exterior wavespeed
WAVESPEED_1 = WAVESPEED/1.2           # Interior wavespeed
LAMBDA_EXT = 4e-3                     # Exterior wavelength
FREC = WAVESPEED/LAMBDA_EXT           # Exterior frecuency
RHO_0 = 1000                          # Exterior density


INTDENSITYFUN = lambda x, y, z: 1000 - 800*np.sin(2*np.pi*x/SIZE)*np.sin(2*np.pi*y/SIZE)*np.cos(np.pi*z/SIZE)
WAVESPEEDFUN = lambda x, y, z: WAVESPEED_1

DXINTDENSITYFUN = lambda x, y, z: -800*(2*np.pi/SIZE)*np.cos(2*np.pi*x/SIZE)*np.sin(2*np.pi*y/SIZE)*np.cos(np.pi*z/SIZE)
DYINTDENSITYFUN = lambda x, y, z: -800*(2*np.pi/SIZE)*np.sin(2*np.pi*x/SIZE)*np.cos(2*np.pi*y/SIZE)*np.cos(np.pi*z/SIZE)
DZINTDENSITYFUN = lambda x, y, z: 800*(np.pi/SIZE)*np.sin(2*np.pi*x/SIZE)*np.sin(2*np.pi*y/SIZE)*np.sin(np.pi*z/SIZE)

DXEXTDENSITYFUN = lambda x, y, z: 0
DYEXTDENSITYFUN = lambda x, y, z: 0
DZEXTDENSITYFUN = lambda x, y, z: 0

PARAMETERS = {
    "SIZE": SIZE,
    "WAVESPEED": WAVESPEED,
    "WAVESPEED_1": WAVESPEED_1,
    "LAMBDA_EXT": LAMBDA_EXT,
    "FREC": FREC,
    "RHO_0": RHO_0,
    "INTDENSITYFUN": INTDENSITYFUN,
    "WAVESPEEDFUN": WAVESPEEDFUN,
    "DXINTDENSITYFUN": DXINTDENSITYFUN,
    "DYINTDENSITYFUN": DYINTDENSITYFUN,
    "DZINTDENSITYFUN": DZINTDENSITYFUN,
    "DXEXTDENSITYFUN": DXEXTDENSITYFUN,
    "DYEXTDENSITYFUN": DYEXTDENSITYFUN,
    "DZEXTDENSITYFUN": DZEXTDENSITYFUN,
}