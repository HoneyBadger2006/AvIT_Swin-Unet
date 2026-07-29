'''
CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing.

Applied via cv2.createCLAHE() to the L channel of the LAB color space (not
independently per RGB channel) -- equalizing each RGB channel separately
shifts hue/saturation and produces color artifacts, whereas the LAB-L
approach boosts local contrast while leaving color information untouched.
Standard starting parameters (clipLimit=2.0, tileGridSize=(8,8)).
'''
import cv2


def clahe_enhance(img_rgb, clip_limit=2.0, tile_grid_size=(8, 8)):
    '''img_rgb: HxWx3 uint8 array (RGB). Returns contrast-enhanced HxWx3 uint8 array (RGB).'''
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
