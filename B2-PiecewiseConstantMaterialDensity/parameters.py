'''
Module contains simulation related parameters.
'''
SIZE_EXT = 7e-3
SIZE_INT = 5e-3
WAVESPEED = 1482.3                    # Exterior wavespeed
WAVESPEED_1 = WAVESPEED/1.2           # Mid wavespeed
WAVESPEED_2 = WAVESPEED/1.2           # Interior wavespeed
LAMBDA_EXT = 3e-3                     # Exterior wavelength
FREC = WAVESPEED/LAMBDA_EXT           # Exterior frecuency
RHO_0 = 1000                          # Exterior density
RHO_1 = 1000                          # Mid density
RHO_2 = 500                           # Interior density

INTDENSITYFUN = lambda x, y, z: RHO_2
EXTDENSITYFUN = lambda x, y, z: RHO_1
INTWAVESPEEDFUN = lambda x, y, z: WAVESPEED_2
EXTWAVESPEEDFUN = lambda x, y, z: WAVESPEED_1

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
}