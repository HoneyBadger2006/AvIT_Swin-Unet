'''
DullRazor hair removal, ported directly (same ops, same params) from
github.com/BlueDokk/Dullrazor-algorithm/blob/main/dullrazor.py — not a
reimplementation from the README description. Only change: wrapped as a
function operating on an in-memory RGB array (uint8) instead of a script
that reads/displays a single hardcoded file.
'''
import cv2


def dullrazor(img_rgb):
    '''img_rgb: HxWx3 uint8 array (RGB). Returns hair-removed HxWx3 uint8 array (RGB).'''
    grayScale = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1, (9, 9))  # type=1 -> MORPH_CROSS, matches original repo exactly
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
    _, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
    dst = cv2.inpaint(img_rgb, mask, 6, cv2.INPAINT_TELEA)
    return dst
