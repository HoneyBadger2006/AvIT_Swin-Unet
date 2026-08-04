import torch
from tta_inference import TTA_VIEWS, tta_forward


def test_each_view_transform_is_exactly_invertible():
    # inv(fwd(x)) must reconstruct x exactly for every view, on a non-symmetric
    # tensor (arange, not random) so a wrong rotation direction can't hide by luck.
    x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
    for name, fwd, inv in TTA_VIEWS:
        recon = inv(fwd(x))
        assert torch.equal(recon, x), f"{name}: inverse did not reconstruct the original tensor"


def test_rot90_and_rot270_are_distinct_transforms():
    x = torch.arange(1 * 1 * 4 * 4, dtype=torch.float32).reshape(1, 1, 4, 4)
    rot90 = TTA_VIEWS[3][1](x)
    rot270 = TTA_VIEWS[4][1](x)
    assert not torch.equal(rot90, rot270), "rot90 and rot270 views should differ on a non-symmetric input"


def test_tta_forward_with_orientation_invariant_model_matches_single_pass():
    # A model whose output doesn't depend on input orientation (constant map) should,
    # after 5x forward+inverse+average, reduce back to exactly that same constant map -
    # sanity-checks the averaging step doesn't introduce spatial corruption.
    class ConstantModel:
        def __call__(self, x):
            b, _, h, w = x.shape
            return torch.full((b, 1, h, w), 2.0)  # arbitrary logit, same shape

        def eval(self):
            return self

    model = ConstantModel()
    img = torch.randn(2, 3, 8, 8)
    avg_prob, orig_prob = tta_forward(model, 'SwinUnet', img)
    expected = torch.sigmoid(torch.full((2, 1, 8, 8), 2.0))
    assert torch.allclose(avg_prob, expected, atol=1e-6)
    assert torch.allclose(orig_prob, expected, atol=1e-6)


if __name__ == '__main__':
    test_each_view_transform_is_exactly_invertible()
    test_rot90_and_rot270_are_distinct_transforms()
    test_tta_forward_with_orientation_invariant_model_matches_single_pass()
    print('All TTA transform tests passed.')
