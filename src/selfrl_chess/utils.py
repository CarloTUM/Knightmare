import torch
from .network import PolicyValueNet
from .config import INPUT_PLANES, NUM_FILTERS, NUM_BLOCKS, ACTION_SIZE, DEVICE

def test_network_forward():
    model = PolicyValueNet(
        input_planes=INPUT_PLANES,
        num_filters=NUM_FILTERS,
        num_blocks=NUM_BLOCKS,
        action_size=ACTION_SIZE,
    ).to(DEVICE)
    model.eval()

    x = torch.rand(1, INPUT_PLANES, 8, 8, device=DEVICE)
    with torch.no_grad():
        policy_log_probs, value = model(x)

    assert policy_log_probs.shape == (1, ACTION_SIZE)
    assert value.shape == (1, 1)
    print("✅ Network forward test passed!")
