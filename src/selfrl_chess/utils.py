# src/selfrl_chess/utils.py
import chess
import numpy as np

def _build_move_mappings():
    """
    Erzeuge Mapping zwischen UCI-Strings und Netzwerk-Indices.
    Basierend auf 4672 möglichen Zügen: 64 Ausgangsfelder × 73 Richtungen/Distanz-Kombinationen.
    """
    files = 'abcdefgh'
    ranks = '12345678'
    promotions = ['q', 'r', 'b', 'n']
    ucis = []
    # Sliding and knight/king moves as UCI without promotions
    for f1 in files:
        for r1 in ranks:
            for f2 in files:
                for r2 in ranks:
                    # Basis-UCI
                    ucis.append(f1 + r1 + f2 + r2)
                    # Promotions: nur bei Vorstoss von 7. nach 8. Reihe
                    if r1 == '7' and r2 == '8':
                        for promo in promotions:
                            ucis.append(f1 + r1 + f2 + r2 + promo)
    # Nur die ersten 4672 Einträge behalten
    ucis = ucis[:4672]
    uci_to_idx = {uci: idx for idx, uci in enumerate(ucis)}
    idx_to_uci = {idx: uci for uci, idx in uci_to_idx.items()}
    return uci_to_idx, idx_to_uci

# Globale Mapping-Objekte
UCI_TO_IDX, IDX_TO_UCI = _build_move_mappings()


def move_to_index(move: chess.Move) -> int:
    """
    Wandelt einen chess.Move in den entsprechenden Netzwerk-Index um.
    """
    uci = move.uci()
    return UCI_TO_IDX[uci]


def index_to_move(idx: int) -> chess.Move:
    """
    Wandelt einen Netzwerk-Index zurück in einen chess.Move um.
    """
    uci = IDX_TO_UCI[idx]
    return chess.Move.from_uci(uci)


def test_network_forward():
    import torch
    from .network import PolicyValueNet
    from .config import INPUT_PLANES, NUM_FILTERS, NUM_BLOCKS, ACTION_SIZE, DEVICE

    # Modell initialisieren
    model = PolicyValueNet(
        input_planes=INPUT_PLANES,
        num_filters=NUM_FILTERS,
        num_blocks=NUM_BLOCKS,
        action_size=ACTION_SIZE,
    ).to(DEVICE)
    model.eval()

    # Dummy-Eingabe: (1, 17, 8, 8)
    x = torch.rand(1, INPUT_PLANES, 8, 8, device=DEVICE)
    with torch.no_grad():
        policy_log_probs, value = model(x)

    assert policy_log_probs.shape == (1, ACTION_SIZE), (
        f"Expected policy shape (1, {ACTION_SIZE}), got {policy_log_probs.shape}"
    )
    assert value.shape == (1, 1), (
        f"Expected value shape (1, 1), got {value.shape}"
    )
    print("Network forward test passed!")
