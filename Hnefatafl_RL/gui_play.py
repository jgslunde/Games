"""
Interactive Tafl GUI with neural network evaluation display.

Supports Brandubh (7x7), Tablut (9x9), and Hnefatafl (11x11) variants.

Displays:
- Board state
- Raw network policy probabilities OR MCTS-based policy (toggle with 'M')
- Value evaluation from current player's perspective

Controls:
- Click piece to select, click destination to move
- Right-click to deselect
- Press 'M' to toggle between Raw Network and MCTS evaluation
- Press 'V' to toggle value display on/off
- Press 'P' to toggle percentage display on/off
- Press 'A' to make AI move for current player
- Press 'R' to reset game
- Or use clickable buttons in the info panel

Usage:
    python gui_play.py <checkpoint_path> [--game {brandubh,tablut,hnefatafl}] [--simulations N] [--c-puct C]
    
Example:
    python gui_play.py checkpoints/best_model.pth
    python gui_play.py checkpoints/best_model.pth --game tablut
    python gui_play.py checkpoints/best_model.pth --game hnefatafl
    python gui_play.py checkpoints/best_model.pth --simulations 200 --c-puct 1.5
"""

import sys
import argparse
import numpy as np
import torch
import pygame
from typing import Optional, Tuple, List

from brandubh import Brandubh, EMPTY, ATTACKER, DEFENDER, KING
from tablut import Tablut
from hnefatafl import Hnefatafl
from network import BrandubhNet, MoveEncoder
from network_tablut import TablutNet, TablutMoveEncoder
from network_hnefatafl import HnefataflNet, HnefataflMoveEncoder


# Colors - Updated palette for better aesthetics
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
# Board colors - specific hex colors
BOARD_DARK = (181, 136, 99)     # #b58863 brown
BOARD_LIGHT = (233, 195, 135)   # #e9c387 beige
BOARD_BORDER = (70, 50, 30)     # Dark border
# Special squares
CORNER_COLOR = (123, 123, 123)  # #7b7b7b gray
THRONE_COLOR = (123, 123, 123)  # #7b7b7b gray (same as corners)
# Piece colors
ATTACKER_COLOR = (40, 40, 40)   # Dark gray/black
ATTACKER_OUTLINE = (20, 20, 20)
DEFENDER_COLOR = (240, 240, 240) # Off-white
DEFENDER_OUTLINE = (180, 180, 180)
KING_COLOR = (218, 165, 32)      # Goldenrod
KING_OUTLINE = (184, 134, 11)    # Dark goldenrod
# UI colors
PANEL_BG = (45, 52, 62)          # Dark blue-gray
PANEL_ACCENT = (58, 68, 82)      # Lighter blue-gray
TEXT_PRIMARY = (236, 240, 241)   # Off-white text
TEXT_SECONDARY = (149, 165, 166) # Gray text
ACCENT_COLOR = (52, 152, 219)    # Blue accent
SUCCESS_COLOR = (46, 204, 113)   # Green
DANGER_COLOR = (231, 76, 60)     # Red
HIGHLIGHT_COLOR = (52, 152, 219) # Blue highlight
MOVE_HIGHLIGHT = (46, 204, 113, 100) # Semi-transparent green
PROBABILITY_TEXT_COLOR = (138, 43, 226)  # Purple (visible on all backgrounds)

# Board settings
WINDOW_WIDTH = 1350  # 50% wider than original 900
WINDOW_HEIGHT = 1080  # 20% taller than original 900
INFO_PANEL_WIDTH = 450  # Proportionally wider (50% more than 300)
BOARD_AREA_WIDTH = WINDOW_WIDTH - INFO_PANEL_WIDTH
# self.board_size, self.square_size, offsets, and self.piece_radius will be set per instance


class TaflGUI:
    def __init__(self, checkpoint_path: Optional[str] = None, game_type: str = 'brandubh', num_simulations: int = 100, c_puct: float = 1.4,
                 king_capture_pieces: int = 2, king_can_capture: bool = True, 
                 throne_is_hostile: bool = False, throne_enabled: bool = True, force_rules: bool = False):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Store game rules (may be overridden by checkpoint)
        self.game_type = game_type.lower()
        self.king_capture_pieces = king_capture_pieces
        self.king_can_capture = king_can_capture
        self.throne_is_hostile = throne_is_hostile
        self.throne_enabled = throne_enabled
        self.force_rules = force_rules
        
        # Initialize game
        if self.game_type == 'tablut':
            self.game = Tablut(
                king_capture_pieces=king_capture_pieces,
                king_can_capture=king_can_capture,
                throne_is_hostile=throne_is_hostile,
                throne_enabled=throne_enabled
            )
            self.board_size = 9
            game_name = "Tablut"
        elif self.game_type == 'hnefatafl':
            self.game = Hnefatafl(
                king_capture_pieces=king_capture_pieces,
                king_can_capture=king_can_capture,
                throne_is_hostile=throne_is_hostile,
                throne_enabled=throne_enabled
            )
            self.board_size = 11
            game_name = "Hnefatafl"
        else:  # default to brandubh
            self.game = Brandubh(
                king_capture_pieces=king_capture_pieces,
                king_can_capture=king_can_capture,
                throne_is_hostile=throne_is_hostile,
                throne_enabled=throne_enabled
            )
            self.board_size = 7
            game_name = "Brandubh"
        
        # Calculate board drawing constants based on board size
        self.square_size = BOARD_AREA_WIDTH // self.board_size
        self.board_offset_x = (BOARD_AREA_WIDTH - self.square_size * self.board_size) // 2
        self.board_offset_y = (WINDOW_HEIGHT - self.square_size * self.board_size) // 2
        self.piece_radius = self.square_size // 3
        
        # Load neural network (optional)
        self.network = None
        if checkpoint_path:
            self.network = self._load_network(checkpoint_path)
            pygame.display.set_caption(f"{game_name} - AI Evaluation")
        else:
            pygame.display.set_caption(f"{game_name}")
        
        self.clock = pygame.time.Clock()
        
        # MCTS settings
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.mcts = None  # MCTS object for evaluation (created when needed)
        self.ai_agent = None  # Agent object for making AI moves (created when needed)
        
        # Selected piece state
        self.selected_piece = None  # (row, col) of selected piece
        self.legal_moves_from_selected = []  # List of legal moves from selected piece
        
        # Evaluation mode toggle
        self.use_mcts = False  # Toggle between raw network and MCTS evaluation
        
        # Display toggles
        self.show_value = True  # Toggle to show/hide value evaluation
        self.show_percentages = True  # Toggle to show/hide percentages on board
        
        # Network evaluation
        self.policy_probs = None
        self.value = None
        self.piece_selection_probs = None  # Aggregated probabilities per piece
        self.move_probs_from_selected = None  # Probabilities for moves from selected piece
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.font_tiny = pygame.font.Font(None, 18)
        self.font_probability = pygame.font.Font(None, 26)  # Larger for probabilities (was 22)
        
        # Initial evaluation
        if self.network:
            self._evaluate_position()
    
    def _get_move_encoder(self):
        """Get the appropriate MoveEncoder for the current game type."""
        if self.game_type == 'tablut':
            return TablutMoveEncoder
        elif self.game_type == 'hnefatafl':
            return HnefataflMoveEncoder
        else:
            return MoveEncoder
    
    def _get_policy_size(self):
        """Get the policy size for the current game type."""
        if self.game_type == 'tablut':
            return 2592  # 81 * 4 * 8
        elif self.game_type == 'hnefatafl':
            return 4840  # 121 * 4 * 10
        else:
            return 1176  # 49 * 4 * 6
    
    def _load_network(self, checkpoint_path: str):
        """Load neural network from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Extract config from checkpoint
            if 'config' in checkpoint:
                config = checkpoint['config']
                num_res_blocks = config.get('num_res_blocks', 4)
                num_channels = config.get('num_channels', 64)
                
                # Update game rules from checkpoint config if available (unless force_rules is set)
                # This ensures the GUI uses the same rules as training
                if not self.force_rules:
                    self.king_capture_pieces = config.get('king_capture_pieces', self.king_capture_pieces)
                    self.king_can_capture = config.get('king_can_capture', self.king_can_capture)
                    self.throne_is_hostile = config.get('throne_is_hostile', self.throne_is_hostile)
                    self.throne_enabled = config.get('throne_enabled', self.throne_enabled)
                    print("Loaded game rules from checkpoint:")
                else:
                    print("Using command-line rules (--force-rules):")
                
                # Recreate game with correct rules
                if self.game_type == 'tablut':
                    self.game = Tablut(
                        king_capture_pieces=self.king_capture_pieces,
                        king_can_capture=self.king_can_capture,
                        throne_is_hostile=self.throne_is_hostile,
                        throne_enabled=self.throne_enabled
                    )
                elif self.game_type == 'hnefatafl':
                    self.game = Hnefatafl(
                        king_capture_pieces=self.king_capture_pieces,
                        king_can_capture=self.king_can_capture,
                        throne_is_hostile=self.throne_is_hostile,
                        throne_enabled=self.throne_enabled
                    )
                else:
                    self.game = Brandubh(
                        king_capture_pieces=self.king_capture_pieces,
                        king_can_capture=self.king_can_capture,
                        throne_is_hostile=self.throne_is_hostile,
                        throne_enabled=self.throne_enabled
                    )
                
                print(f"  King capture pieces: {self.king_capture_pieces}")
                print(f"  King can capture: {self.king_can_capture}")
                print(f"  Throne is hostile: {self.throne_is_hostile}")
                print(f"  Throne enabled: {self.throne_enabled}")
            else:
                # Default values
                num_res_blocks = 4
                num_channels = 64
                print("Warning: No config found in checkpoint, using command-line/default rules")
            
            # Create and load network (select appropriate network for board size)
            if self.game_type == 'tablut':
                network = TablutNet(num_res_blocks=num_res_blocks, num_channels=num_channels)
            elif self.game_type == 'hnefatafl':
                network = HnefataflNet(num_res_blocks=num_res_blocks, num_channels=num_channels)
            else:
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
        if not self.network:
            return  # No evaluation if no network loaded
        
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
        legal_mask = self._get_move_encoder().get_legal_move_mask(self.game)
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
                           c_puct=self.c_puct, device='cpu',
                           move_encoder_class=self._get_move_encoder())
        
        # Run MCTS search
        visit_probs = self.mcts.search(self.game.clone())
        
        # Convert visit probabilities to policy vector
        self.policy_probs = np.zeros(self._get_policy_size(), dtype=np.float32)
        encoder = self._get_move_encoder()
        for move, prob in visit_probs.items():
            move_idx = encoder.encode_move(move)
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
        if not self.network or self.policy_probs is None:
            self.piece_selection_probs = None
            return
            
        self.piece_selection_probs = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        
        for r in range(self.board_size):
            for c in range(self.board_size):
                piece = self.game.board[r, c]
                
                # Check if this piece belongs to current player
                if self.game.current_player == 0:  # Attacker
                    if piece != ATTACKER:
                        continue
                else:  # Defender
                    if piece not in [DEFENDER, KING]:
                        continue
                
                # Sum probabilities of all moves from this piece
                encoder = self._get_move_encoder()
                total_prob = 0.0
                for move in self.game.get_legal_moves():
                    if move[0] == r and move[1] == c:
                        move_idx = encoder.encode_move(move)
                        total_prob += self.policy_probs[move_idx]
                
                self.piece_selection_probs[r, c] = total_prob
    
    def _compute_move_probs_from_selected(self):
        """Compute move probabilities from selected piece."""
        if self.selected_piece is None or not self.network or self.policy_probs is None:
            self.move_probs_from_selected = None
            return
        
        self.move_probs_from_selected = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        
        from_r, from_c = self.selected_piece
        encoder = self._get_move_encoder()
        for move in self.legal_moves_from_selected:
            to_r, to_c = move[2], move[3]
            move_idx = encoder.encode_move(move)
            self.move_probs_from_selected[to_r, to_c] = self.policy_probs[move_idx]
    
    def _make_ai_move(self):
        """Make the AI select and execute a move for the current player."""
        if self.game.game_over or not self.network:
            return
        
        # Create AI agent if needed (separate from MCTS evaluation object)
        if self.ai_agent is None:
            from agent import Agent
            self.ai_agent = Agent(self.network, num_simulations=self.num_simulations, 
                            c_puct=self.c_puct, device='cpu',
                            add_dirichlet_noise=False,
                            move_encoder_class=self._get_move_encoder())
        
        print(f"AI making move for {'Attackers' if self.game.current_player == 0 else 'Defenders'}...")
        
        # Get AI move (temperature=0 for best move)
        move = self.ai_agent.select_move(self.game, temperature=0.0)
        
        if move is not None:
            from_r, from_c, to_r, to_c = move
            print(f"  AI selected: ({from_r},{from_c}) → ({to_r},{to_c})")
            
            # Make the move
            self.game.make_move(move)
            self.selected_piece = None
            self.legal_moves_from_selected = []
            self.move_probs_from_selected = None
            
            # Re-evaluate position
            if not self.game.game_over:
                self._evaluate_position()
        else:
            print("  No legal moves available!")
    
    def _board_to_screen(self, row: int, col: int) -> Tuple[int, int]:
        """Convert board coordinates to screen coordinates (center of square)."""
        x = self.board_offset_x + col * self.square_size + self.square_size // 2
        y = self.board_offset_y + row * self.square_size + self.square_size // 2
        return x, y
    
    def _screen_to_board(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """Convert screen coordinates to board coordinates."""
        col = (x - self.board_offset_x) // self.square_size
        row = (y - self.board_offset_y) // self.square_size
        
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            return row, col
        return None
    
    def _draw_board(self):
        """Draw the board and pieces."""
        # Background
        self.screen.fill(WHITE)
        
        # Draw squares
        for row in range(self.board_size):
            for col in range(self.board_size):
                x = self.board_offset_x + col * self.square_size
                y = self.board_offset_y + row * self.square_size
                
                # Square color (checkerboard)
                if (row + col) % 2 == 0:
                    color = BOARD_LIGHT
                else:
                    color = BOARD_DARK
                
                # Special squares - dynamically determine based on board size
                center = self.board_size // 2
                # Only show throne if throne is enabled
                if self.throne_enabled and (row, col) == (center, center):  # Throne at center
                    color = THRONE_COLOR
                elif (row, col) in self.game.corners:  # Corners from game
                    color = CORNER_COLOR
                
                pygame.draw.rect(self.screen, color, (x, y, self.square_size, self.square_size))
                pygame.draw.rect(self.screen, BLACK, (x, y, self.square_size, self.square_size), 1)
        
        # Highlight selected piece
        if self.selected_piece is not None:
            row, col = self.selected_piece
            x = self.board_offset_x + col * self.square_size
            y = self.board_offset_y + row * self.square_size
            pygame.draw.rect(self.screen, SUCCESS_COLOR, (x, y, self.square_size, self.square_size), 5)
        
        # Draw policy probabilities overlay (only if network is loaded and percentages are enabled)
        prob_texts_to_draw = []  # Store probability texts to draw after pieces
        if self.network and self.piece_selection_probs is not None and self.show_percentages:
            if self.selected_piece is None:
                # Show piece selection probabilities
                max_prob = np.max(self.piece_selection_probs) if np.max(self.piece_selection_probs) > 0 else 1.0
                for row in range(self.board_size):
                    for col in range(self.board_size):
                        prob = self.piece_selection_probs[row, col]
                        if prob > 0.001:
                            x = self.board_offset_x + col * self.square_size
                            y = self.board_offset_y + row * self.square_size
                            
                            # Draw semi-transparent overlay
                            alpha = int(255 * (prob / max_prob) * 0.5)
                            s = pygame.Surface((self.square_size, self.square_size))
                            s.set_alpha(alpha)
                            s.fill(SUCCESS_COLOR)
                            self.screen.blit(s, (x, y))
                            
                            # Store probability text to draw later (after pieces)
                            prob_texts_to_draw.append((prob, x, y))
            elif self.move_probs_from_selected is not None:
                # Show move destination probabilities
                max_prob = np.max(self.move_probs_from_selected) if np.max(self.move_probs_from_selected) > 0 else 1.0
                for row in range(self.board_size):
                    for col in range(self.board_size):
                        prob = self.move_probs_from_selected[row, col]
                        if prob > 0.001:
                            x = self.board_offset_x + col * self.square_size
                            y = self.board_offset_y + row * self.square_size
                            
                            # Draw semi-transparent overlay
                            alpha = int(255 * (prob / max_prob) * 0.5)
                            s = pygame.Surface((self.square_size, self.square_size))
                            s.set_alpha(alpha)
                            s.fill(SUCCESS_COLOR)
                            self.screen.blit(s, (x, y))
                            
                            # Store probability text to draw later (after pieces)
                            prob_texts_to_draw.append((prob, x, y))
        
        # Draw pieces with 3D effect
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.board[row, col]
                if piece == EMPTY:
                    continue
                
                x, y = self._board_to_screen(row, col)
                
                if piece == ATTACKER:
                    # Dark piece with 3D effect
                    # Base color (dark gray/black)
                    base_color = (60, 60, 60)
                    # Shadow (bottom-right)
                    shadow_color = (30, 30, 30)
                    pygame.draw.circle(self.screen, shadow_color, (x + 2, y + 2), self.piece_radius)
                    # Main piece
                    pygame.draw.circle(self.screen, base_color, (x, y), self.piece_radius)
                    # Highlight (top-left)
                    highlight_color = (100, 100, 100)
                    pygame.draw.circle(self.screen, highlight_color, 
                                     (x - self.piece_radius//4, y - self.piece_radius//4), 
                                     self.piece_radius//3)
                    # Outline
                    pygame.draw.circle(self.screen, (20, 20, 20), (x, y), self.piece_radius, 2)
                    
                elif piece == DEFENDER:
                    # Light piece with 3D effect
                    # Base color (off-white)
                    base_color = (240, 240, 240)
                    # Shadow (bottom-right)
                    shadow_color = (180, 180, 180)
                    pygame.draw.circle(self.screen, shadow_color, (x + 2, y + 2), self.piece_radius)
                    # Main piece
                    pygame.draw.circle(self.screen, base_color, (x, y), self.piece_radius)
                    # Highlight (top-left)
                    highlight_color = (255, 255, 255)
                    pygame.draw.circle(self.screen, highlight_color, 
                                     (x - self.piece_radius//4, y - self.piece_radius//4), 
                                     self.piece_radius//3)
                    # Outline
                    pygame.draw.circle(self.screen, (160, 160, 160), (x, y), self.piece_radius, 2)
                    
                elif piece == KING:
                    # King piece with 3D effect (yellow/gold)
                    # Base color (goldenrod)
                    base_color = (218, 165, 32)
                    # Shadow (bottom-right)
                    shadow_color = (150, 110, 20)
                    pygame.draw.circle(self.screen, shadow_color, (x + 2, y + 2), self.piece_radius)
                    # Main piece
                    pygame.draw.circle(self.screen, base_color, (x, y), self.piece_radius)
                    # Highlight (top-left)
                    highlight_color = (255, 215, 100)
                    pygame.draw.circle(self.screen, highlight_color, 
                                     (x - self.piece_radius//4, y - self.piece_radius//4), 
                                     self.piece_radius//3)
                    # Outline
                    pygame.draw.circle(self.screen, (140, 100, 20), (x, y), self.piece_radius, 2)
        
        # Draw probability texts AFTER pieces so they appear on top
        for prob, x, y in prob_texts_to_draw:
            prob_text = self.font_probability.render(f"{prob*100:.0f}%", True, PROBABILITY_TEXT_COLOR)
            text_rect = prob_text.get_rect(center=(x + self.square_size//2, y + self.square_size//2))
            self.screen.blit(prob_text, text_rect)
    
    def _draw_info_panel(self):
        """Draw information panel on the right side with improved aesthetics."""
        panel_x = BOARD_AREA_WIDTH
        
        # Background
        pygame.draw.rect(self.screen, PANEL_BG, (panel_x, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT))
        
        y_offset = 30
        
        # Title with gradient effect
        if self.network:
            title_text = "AI Analysis"
        else:
            title_text = "Game Info"
        title = self.font_large.render(title_text, True, TEXT_PRIMARY)
        title_rect = title.get_rect(center=(panel_x + INFO_PANEL_WIDTH//2, y_offset))
        self.screen.blit(title, title_rect)
        y_offset += 60
        
        # Divider
        pygame.draw.line(self.screen, PANEL_ACCENT, (panel_x + 20, y_offset), 
                        (panel_x + INFO_PANEL_WIDTH - 20, y_offset), 2)
        y_offset += 30
        
        # Current player section
        player_text = "Attacker" if self.game.current_player == 0 else "Defender"
        player_label = self.font_medium.render("Current Turn:", True, TEXT_SECONDARY)
        self.screen.blit(player_label, (panel_x + 20, y_offset))
        y_offset += 35
        
        # Player indicator with piece
        piece_x = panel_x + INFO_PANEL_WIDTH // 2
        piece_y = y_offset + 25
        radius = 20
        
        # Draw piece shadow
        pygame.draw.circle(self.screen, (0, 0, 0, 50), (piece_x + 2, piece_y + 2), radius)
        
        if self.game.current_player == 0:  # Attacker
            pygame.draw.circle(self.screen, ATTACKER_COLOR, (piece_x, piece_y), radius)
            pygame.draw.circle(self.screen, ATTACKER_OUTLINE, (piece_x, piece_y), radius, 2)
        else:  # Defender
            pygame.draw.circle(self.screen, DEFENDER_COLOR, (piece_x, piece_y), radius)
            pygame.draw.circle(self.screen, DEFENDER_OUTLINE, (piece_x, piece_y), radius, 2)
        
        player_name = self.font_medium.render(player_text, True, TEXT_PRIMARY)
        name_rect = player_name.get_rect(center=(piece_x, piece_y + 40))
        self.screen.blit(player_name, name_rect)
        y_offset += 100
        
        # Network evaluation section (only if network loaded)
        if self.network and self.value is not None and self.show_value:
            pygame.draw.line(self.screen, PANEL_ACCENT, (panel_x + 20, y_offset), 
                            (panel_x + INFO_PANEL_WIDTH - 20, y_offset), 2)
            y_offset += 30
            
            # Evaluation mode
            mode_text = "MCTS" if self.use_mcts else "Raw Network"
            mode_label = self.font_small.render(f"Mode: {mode_text}", True, TEXT_SECONDARY)
            self.screen.blit(mode_label, (panel_x + 20, y_offset))
            y_offset += 40
            
            # Value evaluation
            value_label = self.font_medium.render("Position Eval:", True, TEXT_PRIMARY)
            self.screen.blit(value_label, (panel_x + 20, y_offset))
            y_offset += 35
            
            # Add labels for attacker/defender (above the bar)
            defender_label = self.font_small.render("Defender", True, KING_COLOR)
            attacker_label = self.font_small.render("Attacker", True, ATTACKER_OUTLINE)
            self.screen.blit(defender_label, (panel_x + 20, y_offset))
            attacker_label_rect = attacker_label.get_rect(topright=(panel_x + INFO_PANEL_WIDTH - 20, y_offset))
            self.screen.blit(attacker_label, attacker_label_rect)
            
            y_offset += 25
            
            # Value bar
            bar_width = INFO_PANEL_WIDTH - 40
            bar_height = 35
            bar_x = panel_x + 20
            bar_y = y_offset
            
            # Rounded background bar
            pygame.draw.rect(self.screen, PANEL_ACCENT, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
            
            # Convert value to attacker's perspective (value from network is current player's perspective)
            attacker_value = self.value if self.game.current_player == 0 else -self.value
            
            # Value bar (positive = good for attacker, negative = good for defender)
            value_normalized = (attacker_value + 1) / 2  # Convert from [-1, 1] to [0, 1]
            value_normalized = max(0, min(1, value_normalized))
            
            if value_normalized > 0.5:
                # Good for attacker
                fill_start = bar_x + bar_width // 2
                fill_width = int((value_normalized - 0.5) * 2 * (bar_width // 2))
                if fill_width > 0:
                    # Use dark color for attacker
                    pygame.draw.rect(self.screen, ATTACKER_OUTLINE, (fill_start, bar_y, fill_width, bar_height), border_radius=5)
            else:
                # Good for defender
                fill_width = int((0.5 - value_normalized) * 2 * (bar_width // 2))
                fill_start = bar_x + bar_width // 2 - fill_width
                if fill_width > 0:
                    # Use gold color for defender
                    pygame.draw.rect(self.screen, KING_COLOR, (fill_start, bar_y, fill_width, bar_height), border_radius=5)
            
            # Center marker
            center_x = bar_x + bar_width // 2
            pygame.draw.line(self.screen, TEXT_PRIMARY, (center_x, bar_y), (center_x, bar_y + bar_height), 2)
            
            # Value text (always from attacker's perspective)
            value_text = self.font_medium.render(f"{attacker_value:+.3f}", True, TEXT_PRIMARY)
            text_rect = value_text.get_rect(center=(bar_x + bar_width//2, bar_y + bar_height + 25))
            self.screen.blit(value_text, text_rect)
            y_offset += 80
        
        # Instructions
        pygame.draw.line(self.screen, PANEL_ACCENT, (panel_x + 20, y_offset), 
                        (panel_x + INFO_PANEL_WIDTH - 20, y_offset), 2)
        y_offset += 25
        
        instructions_title = self.font_medium.render("Controls:", True, TEXT_PRIMARY)
        self.screen.blit(instructions_title, (panel_x + 20, y_offset))
        y_offset += 35
        
        instructions = [
            "• Click piece to select",
            "• Click square to move",
            "• Right-click to deselect",
            "",
            "• Press R to reset game",
        ]
        
        if self.network:
            instructions.extend([
                "• Press M to toggle",
                "  MCTS/Raw Network",
                "• Press V to toggle value",
                "• Press P to toggle %",
                "• Press A for AI move",
            ])
        
        for line in instructions:
            text = self.font_small.render(line, True, TEXT_SECONDARY)
            self.screen.blit(text, (panel_x + 25, y_offset))
            y_offset += 28
        
        # Interactive buttons section (only if network loaded and game not over)
        if self.network and not self.game.game_over:
            y_offset += 20
            pygame.draw.line(self.screen, PANEL_ACCENT, (panel_x + 20, y_offset), 
                            (panel_x + INFO_PANEL_WIDTH - 20, y_offset), 2)
            y_offset += 25
            
            buttons_title = self.font_medium.render("AI Controls:", True, TEXT_PRIMARY)
            self.screen.blit(buttons_title, (panel_x + 20, y_offset))
            y_offset += 35
            
            # Make AI Move button
            button_width = INFO_PANEL_WIDTH - 40
            button_height = 45
            button_x = panel_x + 20
            button_y = y_offset
            
            # Store button rect for click detection
            self.ai_move_button = pygame.Rect(button_x, button_y, button_width, button_height)
            
            # Draw button
            button_color = ACCENT_COLOR
            pygame.draw.rect(self.screen, button_color, self.ai_move_button, border_radius=8)
            pygame.draw.rect(self.screen, TEXT_PRIMARY, self.ai_move_button, 2, border_radius=8)
            
            button_text = self.font_medium.render("Make AI Move", True, TEXT_PRIMARY)
            text_rect = button_text.get_rect(center=self.ai_move_button.center)
            self.screen.blit(button_text, text_rect)
            y_offset += button_height + 20
            
            # Toggle buttons in a row
            toggle_button_width = (INFO_PANEL_WIDTH - 50) // 2
            toggle_button_height = 40
            
            # Show Value toggle button
            value_button_x = panel_x + 20
            value_button_y = y_offset
            self.value_toggle_button = pygame.Rect(value_button_x, value_button_y, toggle_button_width, toggle_button_height)
            
            value_bg_color = SUCCESS_COLOR if self.show_value else PANEL_ACCENT
            pygame.draw.rect(self.screen, value_bg_color, self.value_toggle_button, border_radius=6)
            pygame.draw.rect(self.screen, TEXT_PRIMARY, self.value_toggle_button, 2, border_radius=6)
            
            value_text = self.font_small.render("Value", True, TEXT_PRIMARY)
            value_text_rect = value_text.get_rect(center=self.value_toggle_button.center)
            self.screen.blit(value_text, value_text_rect)
            
            # Show Percentages toggle button
            pct_button_x = value_button_x + toggle_button_width + 10
            pct_button_y = y_offset
            self.pct_toggle_button = pygame.Rect(pct_button_x, pct_button_y, toggle_button_width, toggle_button_height)
            
            pct_bg_color = SUCCESS_COLOR if self.show_percentages else PANEL_ACCENT
            pygame.draw.rect(self.screen, pct_bg_color, self.pct_toggle_button, border_radius=6)
            pygame.draw.rect(self.screen, TEXT_PRIMARY, self.pct_toggle_button, 2, border_radius=6)
            
            pct_text = self.font_small.render("Show %", True, TEXT_PRIMARY)
            pct_text_rect = pct_text.get_rect(center=self.pct_toggle_button.center)
            self.screen.blit(pct_text, pct_text_rect)
            y_offset += toggle_button_height + 15
        else:
            # No buttons to track if network not loaded or game over
            self.ai_move_button = None
            self.value_toggle_button = None
            self.pct_toggle_button = None
        
        # Game status
        if self.game.game_over:
            y_offset += 20
            pygame.draw.line(self.screen, PANEL_ACCENT, (panel_x + 20, y_offset), 
                            (panel_x + INFO_PANEL_WIDTH - 20, y_offset), 2)
            y_offset += 30
            
            # Game over banner
            banner_y = y_offset
            banner_height = 80
            pygame.draw.rect(self.screen, PANEL_ACCENT, 
                           (panel_x + 10, banner_y, INFO_PANEL_WIDTH - 20, banner_height), border_radius=10)
            
            status_text = self.font_large.render("GAME OVER", True, DANGER_COLOR)
            text_rect = status_text.get_rect(center=(panel_x + INFO_PANEL_WIDTH//2, banner_y + 25))
            self.screen.blit(status_text, text_rect)
            
            if self.game.winner == 0:
                winner_text = "Attackers Win!"
                winner_color = ATTACKER_OUTLINE
            elif self.game.winner == 1:
                winner_text = "Defenders Win!"
                winner_color = KING_COLOR
            else:
                winner_text = "Draw!"
                winner_color = TEXT_SECONDARY
            
            winner = self.font_medium.render(winner_text, True, winner_color)
            text_rect = winner.get_rect(center=(panel_x + INFO_PANEL_WIDTH//2, banner_y + 55))
            self.screen.blit(winner, text_rect)
    
    def _handle_click(self, pos: Tuple[int, int], button: int):
        """Handle mouse click."""
        if self.game.game_over:
            return
        
        # Check if click is on a button (only left click)
        if button == 1 and self.network:
            # Check AI Move button
            if hasattr(self, 'ai_move_button') and self.ai_move_button and self.ai_move_button.collidepoint(pos):
                self._make_ai_move()
                return
            
            # Check Value toggle button
            if hasattr(self, 'value_toggle_button') and self.value_toggle_button and self.value_toggle_button.collidepoint(pos):
                self.show_value = not self.show_value
                print(f"Value display: {'ON' if self.show_value else 'OFF'}")
                return
            
            # Check Percentage toggle button
            if hasattr(self, 'pct_toggle_button') and self.pct_toggle_button and self.pct_toggle_button.collidepoint(pos):
                self.show_percentages = not self.show_percentages
                print(f"Percentage display: {'ON' if self.show_percentages else 'OFF'}")
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
                        # Reset game - create appropriate game type with same rules
                        if self.board_size == 9:
                            self.game = Tablut(
                                king_capture_pieces=self.king_capture_pieces,
                                king_can_capture=self.king_can_capture,
                                throne_is_hostile=self.throne_is_hostile,
                                throne_enabled=self.throne_enabled
                            )
                        else:
                            self.game = Brandubh(
                                king_capture_pieces=self.king_capture_pieces,
                                king_can_capture=self.king_can_capture,
                                throne_is_hostile=self.throne_is_hostile,
                                throne_enabled=self.throne_enabled
                            )
                        self.selected_piece = None
                        self.legal_moves_from_selected = []
                        self.move_probs_from_selected = None
                        if self.network:
                            self._evaluate_position()
                    elif event.key == pygame.K_m and self.network:
                        # Toggle MCTS mode (only if network loaded)
                        self.use_mcts = not self.use_mcts
                        print(f"Switched to {'MCTS' if self.use_mcts else 'Raw Network'} evaluation")
                        if self.use_mcts:
                            print(f"  Running {self.num_simulations} simulations per evaluation")
                        # Re-evaluate current position with new mode
                        if not self.game.game_over:
                            self._evaluate_position()
                    elif event.key == pygame.K_v and self.network:
                        # Toggle value display
                        self.show_value = not self.show_value
                        print(f"Value display: {'ON' if self.show_value else 'OFF'}")
                    elif event.key == pygame.K_p and self.network:
                        # Toggle percentage display
                        self.show_percentages = not self.show_percentages
                        print(f"Percentage display: {'ON' if self.show_percentages else 'OFF'}")
                    elif event.key == pygame.K_a and self.network and not self.game.game_over:
                        # Make AI move
                        self._make_ai_move()
            
            self._draw_board()
            self._draw_info_panel()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Tafl Game GUI with Optional AI Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint (.pth file). If not provided, plays without AI evaluation.")
    parser.add_argument("--game", type=str, default="brandubh", choices=["brandubh", "tablut", "hnefatafl"],
                       help="Game variant to play: brandubh (7x7), tablut (9x9), or hnefatafl (11x11) (default: brandubh)")
    parser.add_argument("--simulations", type=int, default=100, 
                       help="Number of MCTS simulations/depth for AI evaluation (default: 100)")
    parser.add_argument("--c-puct", type=float, default=1.4,
                       help="MCTS exploration constant (default: 1.4)")
    
    # Game rule arguments (optional - will be overridden by checkpoint config if present)
    parser.add_argument("--king-capture-pieces", type=int, default=None, choices=[2, 3, 4],
                       help="Number of pieces required to capture king: 2, 3, or 4. "
                            "Note: If loading a checkpoint, rules from checkpoint will be used unless --force-rules is set.")
    parser.add_argument("--king-can-capture", action="store_true", dest="king_can_capture_flag", default=None,
                       help="King CAN help capture enemy pieces (overrides checkpoint)")
    parser.add_argument("--king-cannot-capture", action="store_false", dest="king_can_capture_flag",
                       help="King CANNOT help capture enemy pieces (overrides checkpoint)")
    parser.add_argument("--throne-is-hostile", action="store_true", dest="throne_is_hostile_flag", default=None,
                       help="Throne IS hostile for captures (overrides checkpoint)")
    parser.add_argument("--throne-not-hostile", action="store_false", dest="throne_is_hostile_flag",
                       help="Throne is NOT hostile for captures (overrides checkpoint)")
    parser.add_argument("--throne-enabled", action="store_true", dest="throne_enabled_flag", default=None,
                       help="Throne IS enabled - blocks movement (overrides checkpoint)")
    parser.add_argument("--throne-disabled", action="store_false", dest="throne_enabled_flag",
                       help="Throne is DISABLED - center is normal square (overrides checkpoint)")
    parser.add_argument("--force-rules", action="store_true",
                       help="Use command-line rules for any unspecified options (instead of checkpoint defaults)")
    
    args = parser.parse_args()
    
    # Determine final rules to use
    # If force_rules is set, start with defaults and override with any specified values
    # Otherwise, just pass the specified values and let checkpoint fill in the rest
    if args.force_rules:
        # Start with default rules
        king_capture_pieces = 2
        king_can_capture = True
        throne_is_hostile = False
        throne_enabled = True
        
        # Override with any explicitly specified values
        if args.king_capture_pieces is not None:
            king_capture_pieces = args.king_capture_pieces
        if args.king_can_capture_flag is not None:
            king_can_capture = args.king_can_capture_flag
        if args.throne_is_hostile_flag is not None:
            throne_is_hostile = args.throne_is_hostile_flag
        if args.throne_enabled_flag is not None:
            throne_enabled = args.throne_enabled_flag
    else:
        # Use None for unspecified values, let checkpoint provide defaults
        king_capture_pieces = args.king_capture_pieces if args.king_capture_pieces is not None else 2
        king_can_capture = args.king_can_capture_flag if args.king_can_capture_flag is not None else True
        throne_is_hostile = args.throne_is_hostile_flag if args.throne_is_hostile_flag is not None else False
        throne_enabled = args.throne_enabled_flag if args.throne_enabled_flag is not None else True
    
    # Print info about rule precedence
    if args.checkpoint and not args.force_rules:
        print("Loading checkpoint - will use rules from checkpoint config if available")
        print("Use --force-rules to override with command-line arguments")
    
    gui = TaflGUI(args.checkpoint, game_type=args.game, num_simulations=args.simulations, c_puct=args.c_puct,
                  king_capture_pieces=king_capture_pieces, king_can_capture=king_can_capture,
                  throne_is_hostile=throne_is_hostile, throne_enabled=throne_enabled,
                  force_rules=args.force_rules)
    gui.run()


if __name__ == "__main__":
    main()
