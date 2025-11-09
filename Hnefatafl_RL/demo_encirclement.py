"""
Demonstrate the encirclement rule with a realistic game scenario.
"""

import numpy as np
from hnefatafl import Hnefatafl, EMPTY, ATTACKER, DEFENDER, KING

def demo_encirclement_scenario():
    """
    Create a realistic scenario where attackers are close to encircling the king.
    """
    game = Hnefatafl()
    
    print("=" * 70)
    print("ENCIRCLEMENT DEMONSTRATION")
    print("=" * 70)
    print()
    print("Scenario: Attackers have pushed the defenders back toward the center")
    print("and are forming a continuous chain around them.")
    print()
    
    # Clear the board and set up a realistic mid-game position
    game.board = np.zeros((11, 11), dtype=np.int8)
    
    # King and a few remaining defenders in center area
    game.board[5, 5] = KING
    game.board[5, 4] = DEFENDER
    game.board[6, 5] = DEFENDER
    game.board[4, 5] = DEFENDER
    
    # Attackers forming a nearly complete ring
    # This creates a ring with one gap
    attacker_ring = [
        (3, 3), (3, 4), (3, 5), (3, 6), (3, 7),
        (4, 3),                         (4, 7),
        (5, 3),                         (5, 7),
        (6, 3),                         (6, 7),
        (7, 3), (7, 4), (7, 5),         (7, 7),  # Gap at (7, 6)
    ]
    
    for r, c in attacker_ring:
        game.board[r, c] = ATTACKER
    
    print("BEFORE: Ring incomplete - one gap at position (7, 6)")
    print(game)
    print(f"King encircled: {game._is_king_encircled()}")
    print(f"Game over: {game.game_over}")
    print()
    print("-" * 70)
    print()
    
    # Now close the ring
    print("AFTER: Attacker moves to (7, 6), completing the encirclement")
    game.board[7, 6] = ATTACKER
    print(game)
    
    # Check encirclement
    is_encircled = game._is_king_encircled()
    print(f"King encircled: {is_encircled}")
    
    # This should trigger game over
    game._check_game_over()
    print(f"Game over: {game.game_over}")
    if game.game_over:
        print(f"Winner: {'Attackers (○)' if game.winner == 0 else 'Defenders (●)'}")
    print()
    print("=" * 70)
    print()
    
    # Show the flood-fill visually
    print("FLOOD-FILL VISUALIZATION")
    print("Showing which squares the king's group can reach:")
    print()
    
    # Perform flood fill and track visited squares
    king_pos = (5, 5)
    visited = set()
    stack = [king_pos]
    visited.add(king_pos)
    
    while stack:
        r, c = stack.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < 11 and 0 <= nc < 11):
                continue
            if (nr, nc) in visited:
                continue
            if game.board[nr, nc] != ATTACKER:
                visited.add((nr, nc))
                stack.append((nr, nc))
    
    # Print board with visited squares marked
    print("   0 1 2 3 4 5 6 7 8 9 A")
    print("   ─────────────────────")
    
    symbols = {EMPTY: '·', ATTACKER: '○', DEFENDER: '●', KING: '♔'}
    corners = [(0, 0), (0, 10), (10, 0), (10, 10)]
    
    for r in range(11):
        if r < 10:
            row_str = f"{r}│"
        else:
            row_str = "A│"
        
        for c in range(11):
            if (r, c) in visited:
                if game.board[r, c] == EMPTY:
                    row_str += "█ "  # Reachable empty square
                else:
                    row_str += symbols[game.board[r, c]] + " "
            elif (r, c) in corners:
                row_str += "X "
            else:
                row_str += symbols[game.board[r, c]] + " "
        
        print(row_str)
    
    print()
    print("Legend:")
    print("  ○ = Attacker")
    print("  ● = Defender")
    print("  ♔ = King")
    print("  █ = Reachable empty square (flood-fill reached)")
    print("  · = Unreachable empty square (blocked by attackers)")
    print()
    print(f"Total reachable squares: {len(visited)}")
    print(f"Reached edge: {any(r == 0 or r == 10 or c == 0 or c == 10 for r, c in visited)}")
    print()
    print("=" * 70)

if __name__ == "__main__":
    demo_encirclement_scenario()
