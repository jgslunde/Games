import sys
import numpy as np
import torch

# Optional: soft import pygame with a friendly error message
try:
    import pygame
except ImportError:
    print("This program requires the 'pygame' package. Install it with: pip install pygame")
    raise

from TicTacToe import TicTacToeGame
from TTTnet import TicTacToeNet
from MCTS import MCTS

# -----------------------------
# Simple Tic-Tac-Toe GUI with policy heatmap
# - Human plays both sides; click to place moves for X and O
# - For the active player, shows either:
#   * NN Policy: raw network output (single forward pass, fast)
#   * MCTS Policy: visit-count distribution after tree search (slower, stronger)
# - Background shade indicates relative preference (brighter = higher prob)
# Controls:
#   - Left click: play move for the side to move
#   - R: reset game
#   - M: toggle between NN policy and MCTS policy
#   - L: load checkpoint (opens simple prompt in terminal)
# -----------------------------

CELL_SIZE = 120
GAP = 4
GRID = 3
PAD = 12
WIDTH = GRID * CELL_SIZE + (GRID - 1) * GAP + PAD * 2
HEIGHT = WIDTH + 80  # extra space for status text
BG_COLOR = (245, 245, 245)
LINE_COLOR = (80, 80, 80)
X_COLOR = (30, 30, 30)
O_COLOR = (30, 30, 30)
TEXT_COLOR = (25, 25, 25)
PROB_TEXT_COLOR = (40, 40, 40)

# MCTS settings (raw distribution over visits; no exploration noise, temperature=1)
MCTS_SIMS = 200
C_PUCT = 2.0

FONT_NAME = None  # default pygame font


def mcts_policy_for_game(game: TicTacToeGame, net: TicTacToeNet, num_simulations: int = MCTS_SIMS, c_puct: float = C_PUCT):
    """
    Compute the MCTS visit-based policy distribution for the current position.
    Returns: (probs: np.ndarray shape [9], value: float)

    Notes:
    - This uses temperature=1.0 (raw normalized visit counts), no added exploration noise.
    - Value is shown from the network's value head for reference.
    """
    net.eval()
    # Run MCTS to get visit distribution
    mcts = MCTS(net, num_simulations=num_simulations, c_puct=c_puct)
    probs = mcts.search(game)

    # Also get network value for info line
    with torch.no_grad():
        board_tensor = game.get_nn_input()
        _, value_output = net(board_tensor)
        value = float(value_output.squeeze(0).cpu().numpy())
    return probs, value


def nn_policy_for_game(game: TicTacToeGame, net: TicTacToeNet):
    """
    Compute the raw neural network policy (masked and renormalized).
    Returns: (probs: np.ndarray shape [9], value: float)

    Notes:
    - Single forward pass through the network (fast).
    - Policy is masked to legal moves and renormalized.
    - No tree search or visit counts.
    """
    net.eval()
    with torch.no_grad():
        board_tensor = game.get_nn_input()
        policy_output, value_output = net(board_tensor)

        legal_moves = game.get_legal_moves()
        policy_probs = policy_output.squeeze(0).cpu().numpy()

        legal_mask = np.zeros(9, dtype=np.float32)
        legal_mask[legal_moves] = 1.0

        masked_policy = policy_probs * legal_mask
        s = masked_policy.sum()
        if s > 0:
            masked_policy = masked_policy / s
        else:
            # Fallback to uniform over legal moves
            if len(legal_moves) > 0:
                masked_policy[legal_moves] = 1.0 / len(legal_moves)
        value = float(value_output.squeeze(0).cpu().numpy())
        return masked_policy, value


def draw_board(surface, game: TicTacToeGame, font, prob_font, policy: np.ndarray | None, mode: str):
    surface.fill(BG_COLOR)

    # Title/status space
    title_rect = pygame.Rect(0, 0, WIDTH, 60)
    pygame.draw.rect(surface, BG_COLOR, title_rect)

    # Show policy mode in top-right corner
    mode_text = f"Mode: {mode}"
    mode_surf = font.render(mode_text, True, (80, 80, 180))
    surface.blit(mode_surf, (WIDTH - mode_surf.get_width() - PAD, PAD))

    # Grid origin
    origin_x = PAD
    origin_y = PAD + 60

    # Draw cells with heatmap based on policy
    # Normalize policy for color scaling among empty squares only
    probs = policy if policy is not None else np.zeros(9, dtype=np.float32)
    legal = (game.board == 0)
    max_p = probs[legal].max() if legal.any() else 1.0
    max_p = max(max_p, 1e-8)

    for r in range(GRID):
        for c in range(GRID):
            i = r * GRID + c
            x = origin_x + c * (CELL_SIZE + GAP)
            y = origin_y + r * (CELL_SIZE + GAP)
            cell_rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

            # Background heatmap: only for empty cells
            if game.board[i] == 0 and policy is not None:
                p = probs[i] / max_p
                # Map to greenish tint (higher p -> brighter)
                g = int(220 * p + 20)
                g = max(40, min(240, g))
                color = (255 - g, 255, 255 - g)
                pygame.draw.rect(surface, color, cell_rect, border_radius=8)
            else:
                pygame.draw.rect(surface, (230, 230, 230), cell_rect, border_radius=8)

            # Border
            pygame.draw.rect(surface, LINE_COLOR, cell_rect, width=2, border_radius=8)

            # Draw X/O marks
            center = (x + CELL_SIZE // 2, y + CELL_SIZE // 2)
            if game.board[i] == 1:
                draw_x(surface, center, CELL_SIZE // 2 - 16)
            elif game.board[i] == -1:
                draw_o(surface, center, CELL_SIZE // 2 - 18)
            elif policy is not None and probs[i] > 0:
                # Show probability text in empty cells
                txt = f"{probs[i]:.2f}"
                text_surf = prob_font.render(txt, True, PROB_TEXT_COLOR)
                text_rect = text_surf.get_rect()
                text_rect.bottomright = (x + CELL_SIZE - 6, y + CELL_SIZE - 6)
                surface.blit(text_surf, text_rect)


def draw_x(surface, center, radius):
    x, y = center
    off = radius
    pygame.draw.line(surface, X_COLOR, (x - off, y - off), (x + off, y + off), 5)
    pygame.draw.line(surface, X_COLOR, (x + off, y - off), (x - off, y + off), 5)


def draw_o(surface, center, radius):
    x, y = center
    pygame.draw.circle(surface, O_COLOR, (x, y), radius, 5)


def draw_status(surface, font, game: TicTacToeGame, value: float | None):
    status_lines = []
    player_sym = 'X' if game.current_player == 1 else 'O'

    winner = game.check_winner()
    if winner is None:
        status_lines.append(f"Turn: {player_sym}")
    elif winner == 0:
        status_lines.append("Game Over: Draw")
    elif winner == 1:
        status_lines.append("Game Over: X wins")
    elif winner == -1:
        status_lines.append("Game Over: O wins")

    if value is not None:
        status_lines.append(f"Net value: {value:+.2f}  (−1=O winning, +1=X winning)")

    status_lines.append("[Click] play | [R] reset | [M] toggle mode | [L] load checkpoint")

    y = 16
    for line in status_lines:
        surf = font.render(line, True, TEXT_COLOR)
        surface.blit(surf, (PAD, y))
        y += surf.get_height() + 4


def load_checkpoint_into(net: TicTacToeNet, path: str) -> bool:
    try:
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        net.load_state_dict(state)
        print(f"Loaded checkpoint: {path}")
        return True
    except Exception as e:
        print(f"Failed to load checkpoint '{path}': {e}")
        return False


def board_key(game: TicTacToeGame):
    # Unique key for caching per-position evaluation
    return (tuple(int(x) for x in game.board.tolist()), int(game.current_player))


def main(checkpoint: str | None = None):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Tic-Tac-Toe – Policy Heatmap')
    clock = pygame.time.Clock()
    font = pygame.font.Font(FONT_NAME, 20)
    prob_font = pygame.font.Font(FONT_NAME, 16)

    # Model
    net = TicTacToeNet()
    if checkpoint:
        load_checkpoint_into(net, checkpoint)

    # Game state
    game = TicTacToeGame()

    # Policy display mode: "NN" or "MCTS"
    policy_mode = "MCTS"

    # Cache for policy/value by position to avoid recomputing MCTS every frame
    cached_key = None
    cached_policy = np.zeros(9, dtype=np.float32)
    cached_value = None
    cached_mode = policy_mode

    running = True
    while running:
        # Compute policy/value for current position only when state or mode changes
        k = board_key(game)
        if k != cached_key or policy_mode != cached_mode:
            if policy_mode == "MCTS":
                policy, value = mcts_policy_for_game(game, net, num_simulations=MCTS_SIMS, c_puct=C_PUCT)
            else:  # "NN"
                policy, value = nn_policy_for_game(game, net)
            cached_policy, cached_value, cached_key, cached_mode = policy, value, k, policy_mode
        else:
            policy, value = cached_policy, cached_value

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Reset
                    game.reset()
                    cached_key = None
                elif event.key == pygame.K_m:  # Toggle policy mode
                    policy_mode = "NN" if policy_mode == "MCTS" else "MCTS"
                    cached_key = None  # Force recompute
                elif event.key == pygame.K_l:  # Load checkpoint via terminal
                    pygame.display.iconify()
                    path = input("Enter checkpoint path (or leave blank to cancel): ").strip()
                    if path:
                        load_checkpoint_into(net, path)
                    pygame.display.set_mode((WIDTH, HEIGHT))
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if game.check_winner() is None:
                    # Map click to cell
                    mx, my = event.pos
                    origin_x = PAD
                    origin_y = PAD + 60
                    for r in range(GRID):
                        for c in range(GRID):
                            x = origin_x + c * (CELL_SIZE + GAP)
                            y = origin_y + r * (CELL_SIZE + GAP)
                            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                            if rect.collidepoint(mx, my):
                                idx = r * GRID + c
                                if game.board[idx] == 0:
                                    game.make_move(idx)
                                    cached_key = None
                                break

        # Redraw
        draw_board(screen, game, font, prob_font, policy, policy_mode)
        draw_status(screen, font, game, value)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    ckpt = None
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    main(ckpt)
