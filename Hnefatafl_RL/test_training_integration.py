"""
Test that the game registry (games.py) gives every variant the right move encoder, board size and
legacy network class. The training workers and the tools look variants up there.
"""

from games import GAMES, game_spec, game_spec_for_board


def test_game_registry():
    """Every variant must map to its own encoder, board size and legacy network class."""
    print("=" * 70)
    print("Testing the game registry for all variants")
    print("=" * 70)
    expected = {
        "brandubh": ("MoveEncoder", 7, "BrandubhNet"),
        "tablut": ("TablutMoveEncoder", 9, "TablutNet"),
        "hnefatafl": ("HnefataflMoveEncoder", 11, "HnefataflNet"),
    }
    assert set(GAMES) == set(expected)
    for name, (encoder_name, board_size, network_name) in expected.items():
        spec = game_spec(name)
        assert spec.board_size == board_size
        assert spec.move_encoder_class.__name__ == encoder_name
        assert spec.legacy_network_class.__name__ == network_name
        assert spec.game_class().board.shape == (board_size, board_size)
        assert game_spec_for_board(board_size) is spec
        assert spec.policy_size == board_size * board_size * 4 * (board_size - 1)
        print(f"  {name}: {board_size}x{board_size}, {encoder_name}, {network_name} ✓")
    try:
        game_spec("chess")
        assert False, "unknown game accepted"
    except ValueError:
        pass
    print("All tests passed! ✓")


if __name__ == "__main__":
    test_game_registry()
