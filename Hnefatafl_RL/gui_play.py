"""
Interactive Tafl board for any rule set of tafl.RULESETS (7x7, 9x9 or 11x11): play by hand, against
a trained network, or watch the network play itself, and see what the network thinks of each
position.

With a checkpoint, the rule set, the network and the search settings are read from the checkpoint's
training config. So a checkpoint opens with the rules it was trained with, and an AI that searches
as in its evaluation games. Command-line options override this. Without a checkpoint the board is
for two human players.

Display:
- Move probabilities on the board, from the raw network policy or from the visit shares of an MCTS
  search (toggle with M). Without a selected piece, each piece shows the summed probability of its
  moves; with a selected piece, each destination shows the probability of that move.
- Value of the position from the attackers' side (+1: attackers win, -1: defenders win)

Controls:
- Left-click a piece to select it, then a highlighted square to move it; right-click deselects
- A: AI move for the side to move     M: raw network / MCTS evaluation
- V: value bar on/off                 P: percentages on/off
- R: reset game
- The buttons in the side panel do the same as A, V, P and M

Usage:
    python gui_play.py [--checkpoint PATH] [--ruleset NAME] [options]

Examples:
    python gui_play.py --ruleset copenhagen
    python gui_play.py --checkpoint checkpoints/sami_tablut_defenders_first/line_6x48_gpool/checkpoint_iter_50.pth \\
        --ai-side attacker
    python gui_play.py --checkpoint checkpoints/brandubh/line_4x32/best_model.pth \\
        --simulations 200 --temperature 0.5 --temperature-mode decay --temperature-decay-moves 20
"""

import argparse
import os
import warnings
from typing import Optional, Tuple

import numpy as np
import pygame
import torch

from features import build_features
from mcts import MCTS
from tafl import EMPTY, ATTACKER, DEFENDER, KING, RULESETS, Tafl, get_rules, move_index
from tafl_network import describe_network, load_network

OWN_PIECES = {0: (ATTACKER,), 1: (DEFENDER, KING)}  # pieces of the attackers (0) and defenders (1)

# Board colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BOARD_DARK = (181, 136, 99)
BOARD_LIGHT = (233, 195, 135)
SPECIAL_SQUARE_COLOR = (123, 123, 123)  # corners, and the throne when it is enabled
# Piece colors: body, shadow, highlight, outline
PIECE_COLORS = {ATTACKER: ((60, 60, 60), (30, 30, 30), (100, 100, 100), (20, 20, 20)),
                DEFENDER: ((240, 240, 240), (180, 180, 180), (255, 255, 255), (160, 160, 160)),
                KING: ((218, 165, 32), (150, 110, 20), (255, 215, 100), (140, 100, 20))}
KING_COLOR = (218, 165, 32)
ATTACKER_BAR_COLOR = (20, 20, 20)
# Panel colors
PANEL_BG = (45, 52, 62)
PANEL_ACCENT = (58, 68, 82)
TEXT_PRIMARY = (236, 240, 241)
TEXT_SECONDARY = (149, 165, 166)
ACCENT_COLOR = (52, 152, 219)
SUCCESS_COLOR = (46, 204, 113)
DANGER_COLOR = (231, 76, 60)
PROBABILITY_TEXT_COLOR = (138, 43, 226)  # purple, readable on every square color
MOVE_FROM_HIGHLIGHT = (255, 200, 100)
MOVE_TO_HIGHLIGHT = (100, 255, 150)

# Window layout
WINDOW_WIDTH = 1350
WINDOW_HEIGHT = 1080
INFO_PANEL_WIDTH = 450
BOARD_AREA_WIDTH = WINDOW_WIDTH - INFO_PANEL_WIDTH

KEY_ACTIONS = {pygame.K_a: 'ai', pygame.K_m: 'mcts', pygame.K_v: 'value', pygame.K_p: 'percent',
               pygame.K_r: 'reset'}


class TaflGUI:
    def __init__(self, checkpoint: Optional[str] = None, ruleset: Optional[str] = None,
                 simulations: Optional[int] = None, c_puct: Optional[float] = None,
                 fpu_reduction: Optional[float] = None, dirichlet_noise: bool = False,
                 temperature: float = 0.0, temperature_mode: str = 'fixed',
                 temperature_threshold: int = 0, temperature_decay_moves: int = 30,
                 max_moves: Optional[int] = None, ai_side: str = 'none'):
        """
        Set up the game, and with a checkpoint also the network and its searches; open the window.

        The arguments are the command-line options of main(). Settings left at None are taken from
        the checkpoint's training config, else from the MCTS defaults:
        - ruleset: the rules the checkpoint was trained with, else Brandubh 2
        - simulations, c_puct, fpu_reduction: those of the checkpoint's evaluation (gating) games
        - max_moves: the training move limit (a draw once more than this many plies are played)
        The search always uses the checkpoint's search_discount and input planes, and Dirichlet
        noise (if enabled) its dirichlet_alpha and dirichlet_epsilon.
        """
        self.network, data = load_network(checkpoint) if checkpoint else (None, {})
        config = data.get('config', {})

        # Rules: the named rule set, else the checkpoint's, else Brandubh 2
        trained_rules = get_rules(config['rules']) if config else None
        self.rules = get_rules(ruleset or trained_rules or 'brandubh')
        self.board_size = self.rules.board_size
        if trained_rules is not None and self.rules != trained_rules:
            print(f"Warning: the network was trained under {trained_rules.name}, not {self.rules.name}")
            if trained_rules.board_size != self.board_size:
                raise ValueError(f"Rule set {self.rules.name} is played on another board than the network")

        # Searches: one for the MCTS evaluation display (never noisy), one for AI moves
        self.analysis_mcts, self.ai_mcts = None, None
        self.simulations = (simulations if simulations is not None
                            else config.get('eval_mcts_sims_attacker', 100))
        self.c_puct = c_puct if c_puct is not None else config.get('eval_c_puct', 1.4)
        self.fpu_reduction = (fpu_reduction if fpu_reduction is not None
                              else config.get('eval_fpu_reduction', -0.5))
        # The network's value targets were discounted per move, so the search discounts the same way
        search_discount = config.get('search_discount', 1.0)
        if checkpoint:
            search = dict(num_simulations=self.simulations, c_puct=self.c_puct,
                          fpu_reduction=self.fpu_reduction, feature_spec=self.network.feature_spec,
                          search_discount=search_discount)
            self.analysis_mcts = MCTS(self.network, **search)
            self.ai_mcts = MCTS(self.network, add_dirichlet_noise=dirichlet_noise,
                                dirichlet_alpha=config.get('dirichlet_alpha', 0.3),
                                dirichlet_epsilon=config.get('dirichlet_epsilon', 0.25), **search)

        self.max_moves = max_moves if max_moves is not None else config.get('max_game_length')
        self.ai_players = {'none': (), 'attacker': (0,), 'defender': (1,), 'both': (0, 1)}[ai_side]
        if self.ai_players and self.network is None:
            raise ValueError("--ai-side needs a --checkpoint")
        self.temperature = temperature
        self.temperature_mode = temperature_mode
        self.temperature_threshold = temperature_threshold
        self.temperature_decay_moves = temperature_decay_moves

        # Window and board geometry
        n = self.board_size
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.title = f"{self.rules.name} {n}x{n}"
        pygame.display.set_caption(self.title + (" - AI evaluation" if self.network else ""))
        self.checkpoint_name = (os.path.join(os.path.basename(os.path.dirname(checkpoint)),
                                             os.path.basename(checkpoint)) if checkpoint else None)
        self.square_size = BOARD_AREA_WIDTH // n
        self.board_x = (BOARD_AREA_WIDTH - self.square_size * n) // 2
        self.board_y = (WINDOW_HEIGHT - self.square_size * n) // 2
        self.piece_radius = self.square_size // 3
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.font_tiny = pygame.font.Font(None, 18)
        self.font_probability = pygame.font.Font(None, 26)

        # Display toggles, and the per-game state (board, selection, animations, evaluation)
        self.use_mcts, self.show_value, self.show_percentages = False, True, True
        self.hovered = None  # board square under the mouse
        self.buttons = {}    # action -> button rectangle, filled while drawing the panel
        self._do('reset')

        print(self.rules.describe())
        print(f"Draw after {self.max_moves} plies" if self.max_moves else "No move limit")
        if self.network is not None:
            print(f"Network: {checkpoint} ({describe_network(self.network)})")
            print(f"AI search: {self.simulations} simulations, c_puct {self.c_puct}, FPU reduction "
                  f"{self.fpu_reduction}, discount {search_discount}, Dirichlet noise "
                  f"{'on' if dirichlet_noise else 'off'}; temperature {temperature} "
                  f"({temperature_mode} mode)")

    def _do(self, action: str):
        """Carry out a key or panel-button action: 'reset', 'ai', 'mcts', 'value' or 'percent'."""
        if action == 'reset':
            self.game = Tafl(self.rules)
            self.selected, self.selected_moves = None, []  # selected square and its legal moves
            self.pending_move, self.animation_progress = None, 0.0  # move sliding across the board
            self.last_move, self.highlight_timer = None, 0  # squares of the last move, fading out
            self.captured = []  # (row, col, piece, fade 0-1) of pieces fading out after a capture
            self.king_left_throne = False  # for temperature mode 'king'
            self._evaluate_position()
        elif self.network is None:
            return
        elif action == 'ai' and not self.game.game_over:
            self._make_ai_move()
        elif action == 'mcts':
            self.use_mcts = not self.use_mcts
            print(f"Evaluation: {'MCTS' if self.use_mcts else 'raw network'}")
            self._evaluate_position()
        elif action == 'value':
            self.show_value = not self.show_value
        elif action == 'percent':
            self.show_percentages = not self.show_percentages

    def _evaluate_position(self):
        """
        Set self.move_probs (legal move -> probability) and self.value (value for the side to move)
        from the raw network or from an MCTS search. Both are empty without a network or after the
        end of the game.
        """
        self.move_probs, self.value = {}, None
        if self.network is None or self.game.game_over:
            return
        if self.use_mcts:
            self.move_probs = self.analysis_mcts.search(self.game)
            self.value = self.analysis_mcts.root.mean_value
        else:
            state = build_features(self.game, self.network.feature_spec)
            with torch.inference_mode():
                policy_logits, value = self.network(torch.from_numpy(state).unsqueeze(0))
            # Softmax over the legal moves only (the same as masking the illegal moves)
            moves = self.game.get_legal_moves()
            logits = policy_logits[0].numpy()[[move_index(move, self.board_size) for move in moves]]
            probs = np.exp(logits - logits.max())
            self.move_probs = dict(zip(moves, probs / probs.sum()))
            self.value = value.item()

    def _current_temperature(self) -> float:
        """Temperature of the next AI move, with the same schedules as self-play in train.py."""
        plies = self.game.move_count
        if self.temperature_mode == 'king':
            return 0.0 if self.king_left_throne else self.temperature
        if self.temperature_mode == 'decay':
            return self.temperature * max(0.0, 1.0 - plies / self.temperature_decay_moves)
        return self.temperature if plies < self.temperature_threshold else 0.0

    def _make_ai_move(self):
        """Search the position with the AI settings and start the chosen move."""
        player, temperature = self.game.current_player, self._current_temperature()
        move = self.ai_mcts.select_move(self.game, temperature)
        value = self.ai_mcts.root.mean_value
        attacker_value = value if player == 0 else -value
        print(f"AI move {self.game.move_count + 1} ({('attackers', 'defenders')[player]}): "
              f"{move[:2]} -> {move[2:]}, value {attacker_value:+.3f} for the attackers, "
              f"temperature {temperature:.2f}")
        self._start_move(move)

    def _start_move(self, move: Tuple[int, int, int, int]):
        """Start the slide animation of `move`; the move is made on the board when it ends."""
        self.pending_move, self.animation_progress = move, 0.0
        self.selected, self.selected_moves = None, []

    def _finish_move(self):
        """Make the animated move, record captures, apply the move limit and re-evaluate."""
        move, self.pending_move = self.pending_move, None
        old_board = self.game.board.copy()
        self.game.make_move(move)
        # Pieces that disappeared, other than the moved one, were captured
        for r, c in zip(*np.nonzero((old_board != EMPTY) & (self.game.board == EMPTY))):
            if (r, c) != move[:2]:
                self.captured.append((int(r), int(c), int(old_board[r, c]), 0.0))
        self.last_move, self.highlight_timer = move, 60
        self.king_left_throne = self.king_left_throne or self.game.board[self.game.throne] != KING
        # As in training: a draw once more than max_moves plies are played (a win on that ply wins)
        if (self.max_moves is not None and self.game.move_count > self.max_moves
                and not self.game.game_over):
            self.game.game_over, self.game.winner = True, None
        self._evaluate_position()

    def _update_animation(self):
        """Advance the move animation, the last-move highlight and the capture fade by one frame."""
        if self.pending_move is not None:
            self.animation_progress += 0.08
            if self.animation_progress >= 1.0:
                self._finish_move()
        self.highlight_timer = max(0, self.highlight_timer - 1)
        self.captured = [(r, c, piece, fade + 0.05) for r, c, piece, fade in self.captured
                         if fade < 1.0]

    def _handle_click(self, pos: Tuple[int, int], button: int):
        """Press a panel button, or select a piece or move it; a right-click deselects."""
        for action, rect in self.buttons.items():
            if button == 1 and rect.collidepoint(pos):
                self._do(action)
                return
        square = self._screen_to_board(pos)
        if square is None or self.game.game_over or button not in (1, 3):
            return
        move = (*self.selected, *square) if self.selected is not None else None
        if button == 1 and move in self.selected_moves:
            self._start_move(move)
        elif button == 1 and self.game.board[square] in OWN_PIECES[self.game.current_player]:
            self.selected = square
            self.selected_moves = [m for m in self.game.get_legal_moves() if m[:2] == square]
        else:
            self.selected, self.selected_moves = None, []

    def _square_rect(self, square: Tuple[int, int]) -> pygame.Rect:
        """Screen rectangle of the board square (row, col)."""
        return pygame.Rect(self.board_x + square[1] * self.square_size,
                           self.board_y + square[0] * self.square_size,
                           self.square_size, self.square_size)

    def _screen_to_board(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Board square (row, col) at screen position `pos`, or None outside the board."""
        row = (pos[1] - self.board_y) // self.square_size
        col = (pos[0] - self.board_x) // self.square_size
        return (row, col) if 0 <= row < self.board_size and 0 <= col < self.board_size else None

    def _shade(self, square: Tuple[int, int], color: Tuple[int, int, int], alpha: int):
        """Blend `color` over a board square, with opacity `alpha` (0-255)."""
        overlay = pygame.Surface((self.square_size, self.square_size))
        overlay.set_alpha(alpha)
        overlay.fill(color)
        self.screen.blit(overlay, self._square_rect(square))

    def _text(self, text: str, font: pygame.font.Font, color: Tuple[int, int, int],
              position: Tuple[int, int], anchor: str = 'topleft'):
        """Draw `text` with its `anchor` point ('topleft', 'center', ...) at `position`."""
        surface = font.render(text, True, color)
        self.screen.blit(surface, surface.get_rect(**{anchor: position}))

    def _draw_piece(self, piece: int, center: Tuple[int, int], alpha: int = 255,
                    radius: Optional[int] = None):
        """Draw a piece as a shaded disc at `center`, with opacity `alpha` (0-255)."""
        body, shadow, highlight, outline = PIECE_COLORS[piece]
        r = radius or self.piece_radius
        # Drawn on a transparent surface with room for the shadow, so the whole piece can fade
        surface = pygame.Surface((2 * r + 4, 2 * r + 4), pygame.SRCALPHA)
        c = r + 1
        pygame.draw.circle(surface, shadow, (c + 2, c + 2), r)
        pygame.draw.circle(surface, body, (c, c), r)
        pygame.draw.circle(surface, highlight, (c - r // 4, c - r // 4), r // 3)
        pygame.draw.circle(surface, outline, (c, c), r, 2)
        surface.set_alpha(alpha)
        self.screen.blit(surface, (center[0] - c, center[1] - c))

    def _draw_board(self):
        """Draw squares, highlights, move probabilities and pieces, with the move animations."""
        n, size, board = self.board_size, self.square_size, self.game.board
        self.screen.fill(WHITE)
        special = set(self.game.corners)
        if self.rules.throne:
            special.add(self.game.throne)
        for r in range(n):
            for c in range(n):
                rect = self._square_rect((r, c))
                color = (SPECIAL_SQUARE_COLOR if (r, c) in special
                         else BOARD_LIGHT if (r + c) % 2 == 0 else BOARD_DARK)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 1)

        if self.highlight_timer > 0:
            self._shade(self.last_move[:2], MOVE_FROM_HIGHLIGHT, min(255, 4 * self.highlight_timer))
            self._shade(self.last_move[2:], MOVE_TO_HIGHLIGHT, min(255, 4 * self.highlight_timer))
        animating = self.pending_move is not None
        if not animating:
            for move in self.selected_moves:
                self._shade(move[2:], (100, 200, 100), 80)
                pygame.draw.circle(self.screen, (80, 180, 80), self._square_rect(move[2:]).center,
                                   size // 6, 3)
            hovered = self.hovered
            if hovered is not None and board[hovered] in OWN_PIECES[self.game.current_player]:
                self._shade(hovered, (255, 255, 150), 60)
                pygame.draw.rect(self.screen, (255, 220, 100), self._square_rect(hovered), 3)
            if self.selected is not None:
                self._shade(self.selected, (100, 150, 255), 90)
                pygame.draw.rect(self.screen, (50, 120, 255), self._square_rect(self.selected), 5)

        # Move probabilities: per piece (summed over its moves), or per destination of the selected
        # piece. The green shade is relative to the largest probability shown.
        shares = {}
        if self.show_percentages and not animating:
            for move, prob in self.move_probs.items():
                if self.selected is None:
                    shares[move[:2]] = shares.get(move[:2], 0.0) + prob
                elif move[:2] == self.selected:
                    shares[move[2:]] = prob
            shares = {square: prob for square, prob in shares.items() if prob > 0.001}
            top = max(shares.values(), default=1.0)
            for square, prob in shares.items():
                self._shade(square, SUCCESS_COLOR, int(127 * prob / top))

        for r in range(n):
            for c in range(n):
                if board[r, c] != EMPTY and not (animating and (r, c) == self.pending_move[:2]):
                    self._draw_piece(board[r, c], self._square_rect((r, c)).center)
        for r, c, piece, fade in self.captured:
            self._draw_piece(piece, self._square_rect((r, c)).center,
                             alpha=max(0, int(255 * (1 - fade))))
        if animating:
            (x0, y0) = self._square_rect(self.pending_move[:2]).center
            (x1, y1) = self._square_rect(self.pending_move[2:]).center
            t = self.animation_progress
            t = t * t * (3 - 2 * t)  # smooth start and stop
            self._draw_piece(board[self.pending_move[:2]],
                             (int(x0 + (x1 - x0) * t), int(y0 + (y1 - y0) * t)))

        # Probability texts last, on a white box so they are readable on top of the pieces
        for square, prob in shares.items():
            text = self.font_probability.render(f"{prob * 100:.0f}%", True, PROBABILITY_TEXT_COLOR)
            rect = text.get_rect(center=self._square_rect(square).center)
            pygame.draw.rect(self.screen, WHITE, rect.inflate(8, 4), border_radius=4)
            self.screen.blit(text, rect)

    def _draw_panel(self):
        """Draw the side panel: variant, side to move, evaluation, controls, buttons and result."""
        x0, middle = BOARD_AREA_WIDTH, BOARD_AREA_WIDTH + INFO_PANEL_WIDTH // 2
        left, right, width = x0 + 20, x0 + INFO_PANEL_WIDTH - 20, INFO_PANEL_WIDTH - 40
        pygame.draw.rect(self.screen, PANEL_BG, (x0, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT))
        self.buttons = {}

        self._text(self.title, self.font_large, TEXT_PRIMARY, (middle, 35), 'center')
        self._text(self.checkpoint_name or "No network loaded", self.font_tiny, TEXT_SECONDARY,
                   (middle, 70), 'center')
        y = 95
        pygame.draw.line(self.screen, PANEL_ACCENT, (left, y), (right, y), 2)

        # Side to move and number of plies played
        player = self.game.current_player
        name = ("Attackers", "Defenders")[player] + (" (AI)" if player in self.ai_players else "")
        self._text("Current turn:", self.font_medium, TEXT_SECONDARY, (left, y + 20))
        self._draw_piece((ATTACKER, DEFENDER)[player], (left + 20, y + 80), radius=20)
        self._text(name, self.font_medium, TEXT_PRIMARY, (left + 55, y + 80), 'midleft')
        limit = f" (draw after {self.max_moves})" if self.max_moves else ""
        self._text(f"Moves played: {self.game.move_count}{limit}", self.font_small,
                   TEXT_SECONDARY, (left, y + 115))
        y += 150

        # Value bar from the attackers' side: filled to the right for the attackers, to the left
        # for the defenders
        if self.show_value and self.value is not None:
            pygame.draw.line(self.screen, PANEL_ACCENT, (left, y), (right, y), 2)
            mode = f"MCTS, {self.simulations} simulations" if self.use_mcts else "raw network"
            self._text(f"Position eval ({mode}):", self.font_small, TEXT_SECONDARY, (left, y + 20))
            self._text("Defenders", self.font_small, KING_COLOR, (left, y + 50))
            self._text("Attackers", self.font_small, TEXT_PRIMARY, (right, y + 50), 'topright')
            attacker_value = self.value if player == 0 else -self.value
            bar = pygame.Rect(left, y + 75, width, 35)
            fill = int(min(1.0, abs(attacker_value)) * width / 2)
            color, fill_x = ((ATTACKER_BAR_COLOR, bar.centerx) if attacker_value > 0
                             else (KING_COLOR, bar.centerx - fill))
            pygame.draw.rect(self.screen, PANEL_ACCENT, bar, border_radius=5)
            pygame.draw.rect(self.screen, color, (fill_x, bar.y, fill, bar.height), border_radius=5)
            pygame.draw.line(self.screen, TEXT_PRIMARY, bar.midtop, bar.midbottom, 2)
            self._text(f"{attacker_value:+.3f}", self.font_medium, TEXT_PRIMARY,
                       (bar.centerx, bar.bottom + 22), 'center')
            y += 160

        pygame.draw.line(self.screen, PANEL_ACCENT, (left, y), (right, y), 2)
        self._text("Controls:", self.font_medium, TEXT_PRIMARY, (left, y + 20))
        y += 55
        lines = ["Click a piece, then a green square", "Right-click to deselect", "R: reset game"]
        if self.network is not None:
            lines += ["A: AI move for the side to move", "M: raw network / MCTS evaluation",
                      "V: value bar on/off", "P: percentages on/off"]
        for line in lines:
            self._text(line, self.font_small, TEXT_SECONDARY, (left + 5, y))
            y += 28

        if self.network is not None and not self.game.game_over:
            y += 15
            toggle_width = (width - 20) // 3
            buttons = [('ai', "Make AI Move", True, pygame.Rect(left, y, width, 45))]
            for i, (action, label, on) in enumerate([('value', "Value", self.show_value),
                                                     ('percent', "Show %", self.show_percentages),
                                                     ('mcts', "MCTS", self.use_mcts)]):
                rect = pygame.Rect(left + i * (toggle_width + 10), y + 60, toggle_width, 40)
                buttons.append((action, label, on, rect))
            for action, label, on, rect in buttons:
                color = ACCENT_COLOR if action == 'ai' else SUCCESS_COLOR if on else PANEL_ACCENT
                pygame.draw.rect(self.screen, color, rect, border_radius=8)
                pygame.draw.rect(self.screen, TEXT_PRIMARY, rect, 2, border_radius=8)
                font = self.font_medium if action == 'ai' else self.font_small
                self._text(label, font, TEXT_PRIMARY, rect.center, 'center')
                self.buttons[action] = rect

        if self.game.game_over:
            banner = pygame.Rect(x0 + 10, y + 20, INFO_PANEL_WIDTH - 20, 80)
            pygame.draw.rect(self.screen, PANEL_ACCENT, banner, border_radius=10)
            self._text("GAME OVER", self.font_large, DANGER_COLOR, (middle, banner.y + 25),
                       'center')
            results = {0: ("Attackers win!", TEXT_PRIMARY), 1: ("Defenders win!", KING_COLOR)}
            result, color = results.get(self.game.winner, ("Draw by move limit", TEXT_SECONDARY))
            self._text(result, self.font_medium, color, (middle, banner.y + 58), 'center')

        if self.network is not None:
            settings = (f"{self.simulations} simulations, c_puct {self.c_puct}, "
                        f"FPU {self.fpu_reduction}, T={self._current_temperature():.2f}")
            self._text(settings, self.font_tiny, TEXT_SECONDARY, (right + 10, WINDOW_HEIGHT - 10),
                       'bottomright')

    def run(self):
        """Main loop: handle input, animate, draw, and let the AI move on its turns."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if self.pending_move is not None:
                    continue  # no input while a move slides across the board
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_click(event.pos, event.button)
                elif event.type == pygame.KEYDOWN and event.key in KEY_ACTIONS:
                    self._do(KEY_ACTIONS[event.key])
            self._update_animation()
            self.hovered = self._screen_to_board(pygame.mouse.get_pos())
            self._draw_board()
            self._draw_panel()
            pygame.display.flip()
            # The AI searches after the frame is shown, so the last move is visible while it thinks
            if (self.pending_move is None and not self.game.game_over
                    and self.game.current_player in self.ai_players):
                self._make_ai_move()
            self.clock.tick(60)


def main():
    # Trained weights with tiny (subnormal) values slow the CPU down 10-100x unless flushed to zero
    torch.set_flush_denormal(True)
    # NumPy warns that subnormals are zero, which is what the line above asks for
    warnings.filterwarnings('ignore', category=UserWarning, module='numpy._core.getlimits')

    parser = argparse.ArgumentParser(
        description="Tafl board with optional network evaluation and AI moves. Options that are "
                    "left out are taken from the checkpoint's training config.")
    parser.add_argument("--checkpoint", help="Checkpoint (.pth) for evaluation and AI moves; "
                                             "without it the board is for two human players")
    parser.add_argument("--ruleset", choices=sorted(RULESETS),
                        help="Rule set (default: the checkpoint's, else brandubh)")
    parser.add_argument("--ai-side", choices=['none', 'attacker', 'defender', 'both'],
                        default='none',
                        help="Side(s) the AI moves for by itself (default: none; press A instead)")
    parser.add_argument("--simulations", type=int,
                        help="MCTS simulations per move (default: as in the checkpoint's "
                             "evaluation games, else 100)")
    parser.add_argument("--c-puct", type=float,
                        help="MCTS exploration constant (default: checkpoint eval_c_puct, "
                             "else 1.4)")
    parser.add_argument("--fpu-reduction", type=float,
                        help="First-play urgency reduction (default: checkpoint "
                             "eval_fpu_reduction, else -0.5)")
    parser.add_argument("--dirichlet-noise", action="store_true",
                        help="Add root noise to the AI searches, so AI games vary")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature of AI moves (0: always the most visited move; "
                             "default: 0)")
    parser.add_argument("--temperature-mode", choices=["fixed", "king", "decay"], default="fixed",
                        help="'fixed': drop to 0 after --temperature-threshold plies, 'king': "
                             "drop to 0 once the king leaves the throne, 'decay': linear decay to "
                             "0 over --temperature-decay-moves plies (default: fixed)")
    parser.add_argument("--temperature-threshold", type=int, default=0,
                        help="Plies with temperature in 'fixed' mode (default: 0)")
    parser.add_argument("--temperature-decay-moves", type=int, default=30,
                        help="Plies of the decay in 'decay' mode (default: 30)")
    parser.add_argument("--max-moves", type=int,
                        help="Draw once more than this many plies are played "
                             "(default: the checkpoint's max_game_length, else no limit)")
    args = parser.parse_args()

    TaflGUI(**vars(args)).run()


if __name__ == "__main__":
    main()
