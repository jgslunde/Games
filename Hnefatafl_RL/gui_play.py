"""
Interactive Brandubh GUI with neural network evaluation display.

Displays:
- Board state
- Raw network policy probabilities OR MCTS-based policy (toggle with 'M')
- Value evaluation from current player's perspective

Controls:
- Click piece to select, click destination to move
- Right-click to deselect
- Press 'M' to toggle between Raw Network and MCTS evaluation
- Press 'R' to reset game

Usage:
    python gui_play.py <checkpoint_path> [--simulations N] [--c-puct C]
    
Example:
    python gui_play.py checkpoints/best_model.pth
    python gui_play.py checkpoints/best_model.pth --simulations 200 --c-puct 1.5
"""

import sys
import argparse
import numpy as np
import torch
import pygame
from typing import Optional, Tuple, List

from brandubh import Brandubh, EMPTY, ATTACKER, DEFENDER, KING
from network import BrandubhNet, MoveEncoder


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (139, 90, 43)
LIGHT_BROWN = (205, 133, 63)
RED = (220, 20, 60)
BLUE = (30, 144, 255)
GOLD = (255, 215, 0)
GREEN = (50, 205, 50)
LIGHT_GREEN = (144, 238, 144)
GRAY = (128, 128, 128)
LIGHT_GRAY = (211, 211, 211)
DARK_ORANGE = (204, 85, 0)  # Darker orange for probability text
BRIGHT_GREEN = (0, 255, 0)  # Bright green for probability text - contrasts with brown

# Board settings
WINDOW_SIZE = 900
BOARD_SIZE = 7
INFO_PANEL_WIDTH = 300
BOARD_AREA_SIZE = WINDOW_SIZE - INFO_PANEL_WIDTH
SQUARE_SIZE = BOARD_AREA_SIZE // BOARD_SIZE
BOARD_OFFSET_X = (BOARD_AREA_SIZE - SQUARE_SIZE * BOARD_SIZE) // 2
BOARD_OFFSET_Y = (WINDOW_SIZE - SQUARE_SIZE * BOARD_SIZE) // 2

# Piece sizes
PIECE_RADIUS = SQUARE_SIZE // 3


class BrandubhGUI:
    def __init__(self, checkpoint_path: str, num_simulations: int = 100, c_puct: float = 1.4):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Brandubh - Neural Network Evaluation")
        self.clock = pygame.time.Clock()
        
        # Load neural network
        self.network = self._load_network(checkpoint_path)
        
        # MCTS settings
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.mcts = None  # Will be created when needed
        
        # Game state
        self.game = Brandubh()
        self.selected_piece = None  # (row, col) of selected piece
        self.legal_moves_from_selected = []  # List of legal moves from selected piece
        
        # Evaluation mode toggle
        self.use_mcts = False  # Toggle between raw network and MCTS evaluation
        
        # Network evaluation
        self.policy_probs = None
        self.value = None
        self.piece_selection_probs = None  # Aggregated probabilities per piece
        self.move_probs_from_selected = None  # Probabilities for moves from selected piece
        
        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        self.font_bold = pygame.font.Font(None, 24)  # Bold-ish font for probabilities
        self.font_bold.set_bold(True)
        
        # Initial evaluation
        self._evaluate_position()
    
    def _load_network(self, checkpoint_path: str) -> BrandubhNet:
        """Load neural network from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Extract config from checkpoint
            if 'config' in checkpoint:
                config = checkpoint['config']
                num_res_blocks = config.get('num_res_blocks', 4)
                num_channels = config.get('num_channels', 64)
            else:
                # Default values
                num_res_blocks = 4
                num_channels = 64
            
            # Create and load network
            network = BrandubhNet(num_res_blocks=num_res_blocks, num_channels=num_channels)
            network.load_state_dict(checkpoint['model_state_dict'])
            network.eval()
            
            print(f"Loaded network from {checkpoint_path}")
            print(f"  Architecture: {num_res_blocks} residual blocks, {num_channels} channels")
            
            return network
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            sys.exit(1)
    
    def _evaluate_position(self):
        """Evaluate current position with neural network or MCTS."""
        if self.use_mcts:
            self._evaluate_with_mcts()
        else:
            self._evaluate_with_network()
    
    def _evaluate_with_network(self):
        """Evaluate current position with raw neural network."""
        state = self.game.get_state()
        state_tensor = torch.from_numpy(state).unsqueeze(0)
        
        with torch.no_grad():
            policy_logits, value = self.network(state_tensor)
        
        policy_logits = policy_logits.cpu().numpy()[0]
        self.value = value.cpu().item()
        
        # Mask illegal moves and convert to probabilities
        legal_mask = MoveEncoder.get_legal_move_mask(self.game)
        policy_logits = policy_logits * legal_mask + (1 - legal_mask) * (-1e8)
        
        # Softmax
        exp_logits = np.exp(policy_logits - np.max(policy_logits))
        self.policy_probs = exp_logits / exp_logits.sum()
        
        # Aggregate probabilities per source piece
        self._compute_piece_selection_probs()
    
    def _evaluate_with_mcts(self):
        """Evaluate current position using MCTS."""
        from mcts import MCTS
        
        # Create MCTS if needed
        if self.mcts is None:
            self.mcts = MCTS(self.network, num_simulations=self.num_simulations, 
                           c_puct=self.c_puct, device='cpu')
        
        # Run MCTS search
        visit_probs = self.mcts.search(self.game.clone())
        
        # Convert visit probabilities to policy vector
        self.policy_probs = np.zeros(1176, dtype=np.float32)
        for move, prob in visit_probs.items():
            move_idx = MoveEncoder.encode_move(move)
            self.policy_probs[move_idx] = prob
        
        # Get value from MCTS root node
        # The value is from the perspective of the player at the root (current player)
        root = self.mcts.root
        if root.visit_count > 0:
            # mean_value at root is already from current player's perspective
            self.value = root.mean_value
        else:
            self.value = 0.0
        
        # Aggregate probabilities per source piece
        self._compute_piece_selection_probs()
    
    def _compute_piece_selection_probs(self):
        """Compute aggregated move probabilities for each piece."""
        self.piece_selection_probs = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = self.game.board[r, c]
                
                # Check if this piece belongs to current player
                if self.game.current_player == 0:  # Attacker
                    if piece != ATTACKER:
                        continue
                else:  # Defender
                    if piece not in [DEFENDER, KING]:
                        continue
                
                # Sum probabilities of all moves from this piece
                total_prob = 0.0
                for move in self.game.get_legal_moves():
                    if move[0] == r and move[1] == c:
                        move_idx = MoveEncoder.encode_move(move)
                        total_prob += self.policy_probs[move_idx]
                
                self.piece_selection_probs[r, c] = total_prob
    
    def _compute_move_probs_from_selected(self):
        """Compute move probabilities from selected piece."""
        if self.selected_piece is None:
            self.move_probs_from_selected = None
            return
        
        self.move_probs_from_selected = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        
        from_r, from_c = self.selected_piece
        for move in self.legal_moves_from_selected:
            to_r, to_c = move[2], move[3]
            move_idx = MoveEncoder.encode_move(move)
            self.move_probs_from_selected[to_r, to_c] = self.policy_probs[move_idx]
    
    def _board_to_screen(self, row: int, col: int) -> Tuple[int, int]:
        """Convert board coordinates to screen coordinates (center of square)."""
        x = BOARD_OFFSET_X + col * SQUARE_SIZE + SQUARE_SIZE // 2
        y = BOARD_OFFSET_Y + row * SQUARE_SIZE + SQUARE_SIZE // 2
        return x, y
    
    def _screen_to_board(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """Convert screen coordinates to board coordinates."""
        col = (x - BOARD_OFFSET_X) // SQUARE_SIZE
        row = (y - BOARD_OFFSET_Y) // SQUARE_SIZE
        
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return row, col
        return None
    
    def _draw_board(self):
        """Draw the board and pieces."""
        # Background
        self.screen.fill(WHITE)
        
        # Draw squares
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                x = BOARD_OFFSET_X + col * SQUARE_SIZE
                y = BOARD_OFFSET_Y + row * SQUARE_SIZE
                
                # Square color (checkerboard)
                if (row + col) % 2 == 0:
                    color = LIGHT_BROWN
                else:
                    color = BROWN
                
                # Special squares
                if (row, col) == (3, 3):  # Throne
                    color = GOLD
                elif (row, col) in [(0, 0), (0, 6), (6, 0), (6, 6)]:  # Corners
                    color = GOLD
                
                pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
                pygame.draw.rect(self.screen, BLACK, (x, y, SQUARE_SIZE, SQUARE_SIZE), 1)
        
        # Highlight selected piece
        if self.selected_piece is not None:
            row, col = self.selected_piece
            x = BOARD_OFFSET_X + col * SQUARE_SIZE
            y = BOARD_OFFSET_Y + row * SQUARE_SIZE
            pygame.draw.rect(self.screen, GREEN, (x, y, SQUARE_SIZE, SQUARE_SIZE), 5)
        
        # Draw policy probabilities overlay
        if self.selected_piece is None:
            # Show piece selection probabilities
            max_prob = np.max(self.piece_selection_probs) if np.max(self.piece_selection_probs) > 0 else 1.0
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    prob = self.piece_selection_probs[row, col]
                    if prob > 0.001:
                        x = BOARD_OFFSET_X + col * SQUARE_SIZE
                        y = BOARD_OFFSET_Y + row * SQUARE_SIZE
                        
                        # Draw semi-transparent overlay
                        alpha = int(255 * (prob / max_prob) * 0.5)
                        s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
                        s.set_alpha(alpha)
                        s.fill(LIGHT_GREEN)
                        self.screen.blit(s, (x, y))
                        
                        # Draw probability text ABOVE the piece
                        prob_text = self.font_bold.render(f"{prob*100:.1f}%", True, BRIGHT_GREEN)
                        text_rect = prob_text.get_rect(center=(x + SQUARE_SIZE//2, y + 12))
                        self.screen.blit(prob_text, text_rect)
        else:
            # Show move destination probabilities
            if self.move_probs_from_selected is not None:
                max_prob = np.max(self.move_probs_from_selected) if np.max(self.move_probs_from_selected) > 0 else 1.0
                for row in range(BOARD_SIZE):
                    for col in range(BOARD_SIZE):
                        prob = self.move_probs_from_selected[row, col]
                        if prob > 0.001:
                            x = BOARD_OFFSET_X + col * SQUARE_SIZE
                            y = BOARD_OFFSET_Y + row * SQUARE_SIZE
                            
                            # Draw semi-transparent overlay
                            alpha = int(255 * (prob / max_prob) * 0.5)
                            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
                            s.set_alpha(alpha)
                            s.fill(LIGHT_GREEN)
                            self.screen.blit(s, (x, y))
                            
                            # Draw probability text ABOVE where piece would be
                            prob_text = self.font_bold.render(f"{prob*100:.1f}%", True, BRIGHT_GREEN)
                            text_rect = prob_text.get_rect(center=(x + SQUARE_SIZE//2, y + 12))
                            self.screen.blit(prob_text, text_rect)
        
        # Draw pieces
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.game.board[row, col]
                if piece == EMPTY:
                    continue
                
                x, y = self._board_to_screen(row, col)
                
                if piece == ATTACKER:
                    pygame.draw.circle(self.screen, BLACK, (x, y), PIECE_RADIUS)
                    pygame.draw.circle(self.screen, WHITE, (x, y), PIECE_RADIUS, 2)
                elif piece == DEFENDER:
                    pygame.draw.circle(self.screen, WHITE, (x, y), PIECE_RADIUS)
                    pygame.draw.circle(self.screen, BLACK, (x, y), PIECE_RADIUS, 2)
                elif piece == KING:
                    pygame.draw.circle(self.screen, GOLD, (x, y), PIECE_RADIUS)
                    pygame.draw.circle(self.screen, BLACK, (x, y), PIECE_RADIUS, 2)
                    # Draw crown
                    crown_points = [
                        (x - PIECE_RADIUS//2, y),
                        (x - PIECE_RADIUS//3, y - PIECE_RADIUS//3),
                        (x, y),
                        (x + PIECE_RADIUS//3, y - PIECE_RADIUS//3),
                        (x + PIECE_RADIUS//2, y)
                    ]
                    pygame.draw.lines(self.screen, BLACK, False, crown_points, 2)
    
    def _draw_info_panel(self):
        """Draw information panel on the right side."""
        panel_x = BOARD_AREA_SIZE
        
        # Background
        pygame.draw.rect(self.screen, LIGHT_GRAY, (panel_x, 0, INFO_PANEL_WIDTH, WINDOW_SIZE))
        pygame.draw.line(self.screen, BLACK, (panel_x, 0), (panel_x, WINDOW_SIZE), 2)
        
        y_offset = 20
        
        # Title
        title = self.font_large.render("Network Eval", True, BLACK)
        self.screen.blit(title, (panel_x + 20, y_offset))
        y_offset += 50
        
        # Evaluation mode
        mode_text = "MCTS" if self.use_mcts else "Raw Network"
        mode_label = self.font_small.render(f"Mode: {mode_text}", True, BLACK)
        self.screen.blit(mode_label, (panel_x + 20, y_offset))
        y_offset += 30
        
        # Current player
        player_text = "Attacker" if self.game.current_player == 0 else "Defender"
        player_color = BLACK if self.game.current_player == 0 else WHITE
        player_label = self.font_medium.render(f"Turn: {player_text}", True, BLACK)
        self.screen.blit(player_label, (panel_x + 20, y_offset))
        
        # Draw colored circle for current player
        pygame.draw.circle(self.screen, player_color, (panel_x + 200, y_offset + 15), 15)
        pygame.draw.circle(self.screen, BLACK, (panel_x + 200, y_offset + 15), 15, 2)
        y_offset += 50
        
        # Value evaluation
        pygame.draw.line(self.screen, BLACK, (panel_x + 20, y_offset), (panel_x + INFO_PANEL_WIDTH - 20, y_offset), 2)
        y_offset += 20
        
        value_label = self.font_medium.render("Position Value:", True, BLACK)
        self.screen.blit(value_label, (panel_x + 20, y_offset))
        y_offset += 35
        
        # Value bar
        bar_width = INFO_PANEL_WIDTH - 60
        bar_height = 30
        bar_x = panel_x + 30
        bar_y = y_offset
        
        # Background bar
        pygame.draw.rect(self.screen, LIGHT_GRAY, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, BLACK, (bar_x, bar_y, bar_width, bar_height), 2)
        
        # Value bar (positive = good for current player)
        value_normalized = (self.value + 1) / 2  # Convert from [-1, 1] to [0, 1]
        value_normalized = max(0, min(1, value_normalized))
        
        if value_normalized > 0.5:
            # Good for current player
            fill_start = bar_x + bar_width // 2
            fill_width = int((value_normalized - 0.5) * 2 * (bar_width // 2))
            pygame.draw.rect(self.screen, GREEN, (fill_start, bar_y, fill_width, bar_height))
        else:
            # Bad for current player
            fill_width = int((0.5 - value_normalized) * 2 * (bar_width // 2))
            fill_start = bar_x + bar_width // 2 - fill_width
            pygame.draw.rect(self.screen, RED, (fill_start, bar_y, fill_width, bar_height))
        
        # Center line
        pygame.draw.line(self.screen, BLACK, (bar_x + bar_width//2, bar_y), (bar_x + bar_width//2, bar_y + bar_height), 2)
        
        # Value text
        value_text = self.font_medium.render(f"{self.value:+.3f}", True, BLACK)
        text_rect = value_text.get_rect(center=(bar_x + bar_width//2, bar_y + bar_height + 20))
        self.screen.blit(value_text, text_rect)
        y_offset += 70
        
        # Instructions
        pygame.draw.line(self.screen, BLACK, (panel_x + 20, y_offset), (panel_x + INFO_PANEL_WIDTH - 20, y_offset), 2)
        y_offset += 20
        
        instructions = [
            "Instructions:",
            "",
            "Click a piece to",
            "select it.",
            "",
            "Green overlay shows",
            "preferred moves.",
            "",
            "Click destination",
            "to move.",
            "",
            "Press 'M' to toggle",
            "MCTS/Raw Network.",
            "",
            "Press 'R' to reset.",
            "",
            "Right-click to",
            "deselect.",
        ]
        
        for line in instructions:
            text = self.font_small.render(line, True, BLACK)
            self.screen.blit(text, (panel_x + 20, y_offset))
            y_offset += 25
        
        # Game status
        if self.game.game_over:
            y_offset += 20
            pygame.draw.line(self.screen, BLACK, (panel_x + 20, y_offset), (panel_x + INFO_PANEL_WIDTH - 20, y_offset), 2)
            y_offset += 20
            
            status_text = self.font_large.render("GAME OVER", True, RED)
            text_rect = status_text.get_rect(center=(panel_x + INFO_PANEL_WIDTH//2, y_offset + 20))
            self.screen.blit(status_text, text_rect)
            y_offset += 60
            
            if self.game.winner == 0:
                winner_text = "Attackers Win!"
            elif self.game.winner == 1:
                winner_text = "Defenders Win!"
            else:
                winner_text = "Draw!"
            
            winner = self.font_medium.render(winner_text, True, BLACK)
            text_rect = winner.get_rect(center=(panel_x + INFO_PANEL_WIDTH//2, y_offset))
            self.screen.blit(winner, text_rect)
    
    def _handle_click(self, pos: Tuple[int, int], button: int):
        """Handle mouse click."""
        if self.game.game_over:
            return
        
        board_pos = self._screen_to_board(pos[0], pos[1])
        if board_pos is None:
            return
        
        row, col = board_pos
        
        # Right click - deselect
        if button == 3:
            self.selected_piece = None
            self.legal_moves_from_selected = []
            self.move_probs_from_selected = None
            return
        
        # Left click
        if button == 1:
            if self.selected_piece is None:
                # Select a piece
                piece = self.game.board[row, col]
                
                # Check if piece belongs to current player
                if self.game.current_player == 0 and piece == ATTACKER:
                    self.selected_piece = (row, col)
                    # Get legal moves from this piece
                    self.legal_moves_from_selected = [
                        move for move in self.game.get_legal_moves()
                        if move[0] == row and move[1] == col
                    ]
                    self._compute_move_probs_from_selected()
                elif self.game.current_player == 1 and piece in [DEFENDER, KING]:
                    self.selected_piece = (row, col)
                    # Get legal moves from this piece
                    self.legal_moves_from_selected = [
                        move for move in self.game.get_legal_moves()
                        if move[0] == row and move[1] == col
                    ]
                    self._compute_move_probs_from_selected()
            else:
                # Try to make a move
                from_r, from_c = self.selected_piece
                move = (from_r, from_c, row, col)
                
                if move in self.legal_moves_from_selected:
                    # Make the move
                    self.game.make_move(move)
                    self.selected_piece = None
                    self.legal_moves_from_selected = []
                    self.move_probs_from_selected = None
                    
                    # Re-evaluate position
                    if not self.game.game_over:
                        self._evaluate_position()
                else:
                    # Invalid move - try selecting new piece
                    self.selected_piece = None
                    self.legal_moves_from_selected = []
                    self.move_probs_from_selected = None
                    self._handle_click(pos, button)
    
    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_click(event.pos, event.button)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        # Reset game
                        self.game = Brandubh()
                        self.selected_piece = None
                        self.legal_moves_from_selected = []
                        self.move_probs_from_selected = None
                        self._evaluate_position()
                    elif event.key == pygame.K_m:
                        # Toggle MCTS mode
                        self.use_mcts = not self.use_mcts
                        print(f"Switched to {'MCTS' if self.use_mcts else 'Raw Network'} evaluation")
                        if self.use_mcts:
                            print(f"  Running {self.num_simulations} simulations per evaluation")
                        # Re-evaluate current position with new mode
                        if not self.game.game_over:
                            self._evaluate_position()
            
            self._draw_board()
            self._draw_info_panel()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Brandubh GUI with Neural Network Evaluation")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint (.pth file)")
    parser.add_argument("--simulations", type=int, default=100, 
                       help="Number of MCTS simulations (default: 100)")
    parser.add_argument("--c-puct", type=float, default=1.4,
                       help="MCTS exploration constant (default: 1.4)")
    
    args = parser.parse_args()
    
    gui = BrandubhGUI(args.checkpoint, num_simulations=args.simulations, c_puct=args.c_puct)
    gui.run()


if __name__ == "__main__":
    main()
