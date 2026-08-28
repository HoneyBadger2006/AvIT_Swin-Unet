"""
Builds a single slide-ready composite image (3x3 grid, ~16:9 overall) showing one
representative example per visual failure category from the 43-image cross-network-
overlap qualitative review. Each cell shows (original / ground truth / prediction)
cropped from the existing 4-panel per-image visualizations in both_networks_overlap/
(the 4th "error map" panel is dropped for slide brevity), with a two-line category
banner (name + count) above.

Top row = the 3 priority categories (fuzzy-halo, ink confusion, topology) with a red
border / pink banner. Remaining 6 categories fill rows 2-3 with a neutral gray banner.
"""
import os
from PIL import Image, ImageDraw, ImageFont

SRC_DIR = '../per_image_analysis_v2/low_dice_visualizations/both_networks_overlap'
OUT_PATH = '../per_image_analysis_v2/overlap_category_grid.png'

# (filename, line1, line2, priority)
CELLS = [
    ('avit_0013600_test_dice0.388.png',
     'Fuzzy / Ambiguous Border (Diffuse Halo)', '13/43 images \u2014 largest category', True),
    ('avit_0013835_train_dice0.040.png',
     'Ink Marking Confusion', '3/43 images \u2014 new finding', True),
    ('avit_0015353_test_dice0.252.png',
     'Unusual Topology (Donut-Shaped Lesion)', '2/43 images \u2014 structural limit', True),
    ('avit_0012086_test_dice0.214.png',
     'Low Contrast', '10/43 images', False),
    ('avit_0012448_test_dice0.209.png',
     'Very Small Lesion + Decoy Ring', '4/43 images', False),
    ('avit_0016034_test_dice0.435.png',
     'Cluttered Background / Similar Distractors', '4/43 images', False),
    ('avit_0012388_test_dice0.113.png',
     'Hair Occlusion', '3/43 images', False),
    ('avit_0012265_test_dice0.379.png',
     'Frame-Cropped / Giant Lesion', '2/43 images', False),
    ('avit_0015251_test_dice0.051.png',
     'Likely Ground-Truth Annotation Error', '2/43 images', False),
]

CANVAS_W, CANVAS_H = 1920, 1080
TITLE_H = 55
MARGIN = 20
GAP = 14
BANNER_H = 54
PRIORITY_BORDER = (220, 30, 30)
NORMAL_BORDER = (70, 70, 70)


def load_font(size, bold=False):
    candidates = [
        'C:/Windows/Fonts/segoeuib.ttf' if bold else 'C:/Windows/Fonts/segoeui.ttf',
        'C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',
    ]
    for c in candidates:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


font_title = load_font(28, bold=True)
font_l1 = load_font(17, bold=True)
font_l2 = load_font(14, bold=False)

grid_w = CANVAS_W - 2 * MARGIN - 2 * GAP
grid_h = CANVAS_H - TITLE_H - 2 * MARGIN - 2 * GAP
cell_w = grid_w // 3
cell_h = grid_h // 3
image_area_w = cell_w
image_area_h = cell_h - BANNER_H

canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

title = 'Cross-Network Overlap (43 images) \u2014 Visual Failure Categories, AViT examples'
tw = draw.textlength(title, font=font_title)
draw.text(((CANVAS_W - tw) / 2, 14), title, fill=(0, 0, 0), font=font_title)

for idx, (fname, line1, line2, priority) in enumerate(CELLS):
    row, col = divmod(idx, 3)
    cell_x = MARGIN + col * (cell_w + GAP)
    cell_y = TITLE_H + MARGIN + row * (cell_h + GAP)

    border_color = PRIORITY_BORDER if priority else NORMAL_BORDER
    border_w = 4 if priority else 2

    # banner
    draw.rectangle([cell_x, cell_y, cell_x + cell_w - 1, cell_y + BANNER_H - 1],
                    fill=(255, 225, 225) if priority else (232, 232, 232))
    prefix = 'PRIORITY: ' if priority else ''
    draw.text((cell_x + 8, cell_y + 5), prefix + line1, fill=(0, 0, 0), font=font_l1)
    draw.text((cell_x + 8, cell_y + 28), line2, fill=(40, 40, 40), font=font_l2)

    # image (original/GT/prediction crop), scaled to fit image_area preserving aspect
    im = Image.open(os.path.join(SRC_DIR, fname)).convert('RGB')
    w, h = im.size
    cropped = im.crop((0, 0, int(w * 0.75), h))
    cw, ch = cropped.size
    scale = min(image_area_w / cw, image_area_h / ch)
    new_size = (int(cw * scale), int(ch * scale))
    resized = cropped.resize(new_size, Image.LANCZOS)

    img_x = cell_x + (image_area_w - new_size[0]) // 2
    img_y = cell_y + BANNER_H + (image_area_h - new_size[1]) // 2
    canvas.paste(resized, (img_x, img_y))

    draw.rectangle([cell_x, cell_y, cell_x + cell_w - 1, cell_y + cell_h - 1],
                    outline=border_color, width=border_w)

canvas.save(OUT_PATH)
print('Saved:', OUT_PATH, canvas.size, 'aspect ratio: {:.3f}'.format(CANVAS_W / CANVAS_H))
