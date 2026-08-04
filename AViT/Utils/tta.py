import torch

# (view name, forward transform applied to the input image, exact inverse transform
# applied to the resulting probability map to bring it back to the original orientation)
TTA_VIEWS = [
    ('original', lambda x: x,                                 lambda p: p),
    ('h_flip',   lambda x: torch.flip(x, dims=[-1]),           lambda p: torch.flip(p, dims=[-1])),
    ('v_flip',   lambda x: torch.flip(x, dims=[-2]),           lambda p: torch.flip(p, dims=[-2])),
    ('rot90',    lambda x: torch.rot90(x, k=1, dims=[-2, -1]), lambda p: torch.rot90(p, k=3, dims=[-2, -1])),
    ('rot270',   lambda x: torch.rot90(x, k=3, dims=[-2, -1]), lambda p: torch.rot90(p, k=1, dims=[-2, -1])),
]


def tta_forward(forward_fn, img):
    """5-view test-time-augmentation soft voting. forward_fn(x) -> raw (pre-sigmoid)
    model logits for input x, same shape as x's batch/spatial dims. Runs the network on
    the original image plus horizontal flip, vertical flip, 90-degree rotation, and
    270-degree rotation; each output is sigmoid-activated then inverse-transformed back
    to the original orientation before averaging, so thresholding happens once on the
    5-view average probability map.

    Returns (avg_prob, orig_prob), both [B,1,H,W] sigmoid probability maps in the
    original image's orientation. avg_prob is the 5-view soft-voted average; orig_prob
    is just the original (non-augmented) pass, for a no-TTA comparison.
    """
    probs = []
    orig_prob = None
    with torch.no_grad():
        for name, fwd, inv in TTA_VIEWS:
            aug_img = fwd(img)
            logits = forward_fn(aug_img)
            prob = torch.sigmoid(logits)
            prob_orig_frame = inv(prob)
            probs.append(prob_orig_frame)
            if name == 'original':
                orig_prob = prob_orig_frame
    avg_prob = torch.stack(probs, dim=0).mean(dim=0)
    return avg_prob, orig_prob
