'''
Module contains simulation related parameters.
'''
SIZE = 6
RADIUS = 2.5                       # Radius of sphere
WAVESPEED = 1.                     # Exterior wavespeed
LAMBDA_EXT = 2                     # Exterior wavelength
LAMBDA_EXT_ERROR = 4            
FREC = WAVESPEED/LAMBDA_EXT        # Exterior frecuency
FREC_ERROR = WAVESPEED/LAMBDA_EXT_ERROR  # Exterior frecuency with error
RHO_0 = 1                          # Exterior density

DENSITYFUN = lambda x, y, z: RHO_0
INTWAVESPEEDFUN = lambda x, y, z: 5/6
EXTWAVESPEEDFUN = lambda x, y, z: WAVESPEED

PARAMETERS = {
    "SIZE": SIZE,
    "RADIUS": RADIUS,
    "WAVESPEED": WAVESPEED,
    "LAMBDA_EXT": LAMBDA_EXT,
    "FREC": FREC,
    "RHO_0": RHO_0,
    "DENSITYFUN": DENSITYFUN,
    "INTWAVESPEEDFUN": INTWAVESPEEDFUN,
    "EXTWAVESPEEDFUN": EXTWAVESPEEDFUN,
    "LAMBDA_EXT_ERROR": LAMBDA_EXT_ERROR,
    "FREC_ERROR": FREC_ERROR,
}
