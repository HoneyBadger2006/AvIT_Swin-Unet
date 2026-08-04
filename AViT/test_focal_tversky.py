import torch
from Utils.losses import focal_tversky_loss, dice_loss

def test_perfect_prediction_is_zero():
    pred = torch.tensor([1.0, 0.0, 1.0, 0.0])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = focal_tversky_loss(pred, target, alpha=0.7, gamma=4.0/3.0)
    assert loss.item() < 1e-3, f"expected ~0, got {loss.item()}"

def test_completely_wrong_prediction_near_one():
    pred = torch.tensor([0.0, 1.0, 0.0, 1.0])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = focal_tversky_loss(pred, target, alpha=0.7, gamma=4.0/3.0)
    assert loss.item() > 0.99, f"expected ~1, got {loss.item()}"

def test_alpha_penalizes_false_negatives_more():
    # under-prediction (misses positives -> FN) vs over-prediction (extra positives -> FP),
    # same magnitude of error; alpha=0.7 > 0.5 should penalize the FN case harder.
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])
    under_pred = torch.tensor([0.0, 1.0, 0.0, 0.0])  # misses one positive -> FN
    over_pred  = torch.tensor([1.0, 1.0, 1.0, 0.0])  # extra positive -> FP
    loss_under = focal_tversky_loss(under_pred, target, alpha=0.7, gamma=4.0/3.0)
    loss_over = focal_tversky_loss(over_pred, target, alpha=0.7, gamma=4.0/3.0)
    assert loss_under.item() > loss_over.item(), (
        f"alpha=0.7 should penalize FN-heavy errors more: FN-loss={loss_under.item()} "
        f"should exceed FP-loss={loss_over.item()}"
    )

def test_gamma_one_reduces_to_plain_tversky():
    pred = torch.tensor([0.8, 0.3, 0.6, 0.1])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    alpha = 0.7
    smooth = 1e-5
    tp = torch.sum(pred * target)
    fp = torch.sum(pred * (1 - target))
    fn = torch.sum((1 - pred) * target)
    tversky_index = (tp + smooth) / (tp + alpha * fn + (1 - alpha) * fp + smooth)
    expected = 1 - tversky_index
    actual = focal_tversky_loss(pred, target, alpha=alpha, gamma=1.0)
    assert torch.allclose(actual, expected, atol=1e-6), f"expected {expected.item()}, got {actual.item()}"

def test_compound_loss_shape_sane():
    # sanity check the compound formula (1*dice + 2*FTL + 0.5*BCE) produces a scalar,
    # finite, non-negative-ish loss on realistic-shaped tensors.
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 16, 16)
    label = (torch.rand(2, 1, 16, 16) > 0.5).float()
    prob = torch.sigmoid(logits)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, label)
    dice = dice_loss(prob, label)
    ftl = focal_tversky_loss(prob, label, alpha=0.7, gamma=4.0/3.0)
    loss = 1.0 * dice + 2.0 * ftl + 0.5 * bce
    assert torch.isfinite(loss), f"non-finite compound loss: {loss}"
    assert loss.item() > 0, f"expected positive loss, got {loss.item()}"

if __name__ == '__main__':
    test_perfect_prediction_is_zero()
    test_completely_wrong_prediction_near_one()
    test_alpha_penalizes_false_negatives_more()
    test_gamma_one_reduces_to_plain_tversky()
    test_compound_loss_shape_sane()
    print("All focal tversky loss tests passed.")
