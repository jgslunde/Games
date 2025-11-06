"""
Brandubh (7x7 Tafl) game implementation.
Efficient representation suitable for neural network training.

Board representation:
- 0: Empty
- 1: Attacker (black)
- 2: Defender (white)
- 3: King

Rules:
- Attackers win by capturing the king
- Defenders win by moving the king to a corner
- Pieces are captured by surrounding them on two opposite sides (custodian capture)
- King needs to be surrounded on all 4 sides (or 3 sides + throne)
- Special squares: throne (center) and corners
"""

import numpy as np
from typing import List, Tuple

# Piece types
EMPTY = 0
ATTACKER = 1
DEFENDER = 2
KING = 3

# Players
ATTACKER_PLAYER = 0
DEFENDER_PLAYER = 1


class Brandubh:
    def __init__(self):
        self.board = np.zeros((7, 7), dtype=np.int8)
        self.current_player = ATTACKER_PLAYER  # Attackers start
        self.game_over = False
        self.winner = None
        
        # Special squares
        self.throne = (3, 3)
        self.corners = [(0, 0), (0, 6), (6, 0), (6, 6)]
        self.corner_set = set(self.corners)
        
        # Position history for repetition detection (3-fold repetition = draw)
        self.position_history = []
        self.move_count = 0
        
        # Track if king has moved off throne
        self.king_has_left_throne = False
        
        self._setup_board()
        self._record_position()
    
    def _setup_board(self):
        """Initialize the starting position for Brandubh."""
        # King at center
        self.board[3, 3] = KING
        
        # Defenders around king
        defenders = [(2, 3), (3, 2), (3, 4), (4, 3)]
        for r, c in defenders:
            self.board[r, c] = DEFENDER
        
        # Attackers on edges
        attackers = [
            # Top
            (0, 3),
            # Right
            (3, 6),
            # Bottom
            (6, 3),
            # Left
            (3, 0),
            # Near corners
            (1, 3), (3, 1), (3, 5), (5, 3)
        ]
        for r, c in attackers:
            self.board[r, c] = ATTACKER
    
    def _get_position_hash(self) -> bytes:
        """Get a hashable representation of the current position."""
        # Include board state and current player in the hash
        return (self.board.tobytes(), self.current_player)
    
    def _record_position(self):
        """Record the current position for repetition detection."""
        self.position_history.append(self._get_position_hash())
    
    def _check_repetition_draw(self) -> bool:
        """Check if the current position has occurred 3 times (draw by repetition)."""
        if len(self.position_history) < 6:  # Need at least 6 moves for 3-fold repetition
            return False
        
        current_pos = self._get_position_hash()
        count = self.position_history.count(current_pos)
        return count >= 3
    
    def get_state(self) -> np.ndarray:
        """
        Get board state representation suitable for neural networks.
        Returns 4 planes: [attackers, defenders, king, current_player_plane]
        """
        state = np.zeros((4, 7, 7), dtype=np.float32)
        state[0] = (self.board == ATTACKER).astype(np.float32)
        state[1] = (self.board == DEFENDER).astype(np.float32)
        state[2] = (self.board == KING).astype(np.float32)
        state[3] = np.full((7, 7), self.current_player, dtype=np.float32)
        return state
    
    def get_legal_moves(self) -> List[Tuple[int, int, int, int]]:
        """
        Get all legal moves for current player.
        Returns list of (from_row, from_col, to_row, to_col)
        Filters out moves that would result in three-fold repetition.
        """
        moves = []
        
        if self.current_player == ATTACKER_PLAYER:
            piece_types = [ATTACKER]
        else:
            piece_types = [DEFENDER, KING]
        
        for r in range(7):
            for c in range(7):
                if self.board[r, c] in piece_types:
                    moves.extend(self._get_piece_moves(r, c))
        
        # Filter out moves that would cause three-fold repetition
        legal_moves = []
        for move in moves:
            if not self._would_cause_repetition(move):
                legal_moves.append(move)
        
        return legal_moves
    
    def _would_cause_repetition(self, move: Tuple[int, int, int, int]) -> bool:
        """
        Check if making this move would result in a position that has already 
        occurred twice (i.e., would be the third occurrence).
        """
        from_r, from_c, to_r, to_c = move
        
        # Temporarily make the move
        piece = self.board[from_r, from_c]
        original_to_piece = self.board[to_r, to_c]
        
        self.board[from_r, from_c] = EMPTY
        self.board[to_r, to_c] = piece
        
        # Get position hash after the move (with switched player)
        # Note: We need to check with the NEXT player's turn
        next_player = 1 - self.current_player
        position_hash = (self.board.tobytes(), next_player)
        
        # Count how many times this position has occurred
        count = self.position_history.count(position_hash)
        
        # Undo the move
        self.board[from_r, from_c] = piece
        self.board[to_r, to_c] = original_to_piece
        
        # If this position has already occurred twice, this move would cause repetition
        return count >= 2
    
    def _get_piece_moves(self, r: int, c: int) -> List[Tuple[int, int, int, int]]:
        """Get all legal moves for a piece at position (r, c)."""
        moves = []
        
        # Move in 4 directions: up, down, left, right
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            
            # Keep moving in this direction until blocked
            while 0 <= nr < 7 and 0 <= nc < 7:
                # Can't move to occupied square
                if self.board[nr, nc] != EMPTY:
                    break
                
                # Only king can move to corners and throne
                is_king = self.board[r, c] == KING
                is_special = (nr, nc) == self.throne or (nr, nc) in self.corner_set
                
                if is_special and not is_king:
                    break
                
                moves.append((r, c, nr, nc))
                nr += dr
                nc += dc
        
        return moves
    
    def make_move(self, move: Tuple[int, int, int, int]) -> bool:
        """
        Make a move. Returns True if successful, False if illegal.
        """
        if self.game_over:
            return False
        
        from_r, from_c, to_r, to_c = move
        
        # Validate move
        if move not in self.get_legal_moves():
            return False
        
        # Move piece
        piece = self.board[from_r, from_c]
        self.board[from_r, from_c] = EMPTY
        self.board[to_r, to_c] = piece
        
        # Track if king has left the throne
        if piece == KING and (from_r, from_c) == self.throne:
            self.king_has_left_throne = True
        
        # Increment move counter
        self.move_count += 1
        
        # Check for captures
        self._check_captures(to_r, to_c)
        
        # Switch player and record position (before checking game over for repetition)
        self.current_player = 1 - self.current_player
        self._record_position()
        
        # Check win conditions (includes repetition check)
        self._check_game_over()
        
        return True
    
    def _check_captures(self, r: int, c: int):
        """Check if the move at (r, c) captures any enemy pieces."""
        piece = self.board[r, c]
        
        # Check all 4 directions for captures
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            self._check_capture_in_direction(r, c, dr, dc, piece)
    
    def _check_capture_in_direction(self, r: int, c: int, dr: int, dc: int, piece: int):
        """Check if a capture occurs in a specific direction."""
        nr, nc = r + dr, c + dc
        
        # Check if there's an adjacent enemy
        if not (0 <= nr < 7 and 0 <= nc < 7):
            return
        
        enemy = self.board[nr, nc]
        if enemy == EMPTY or self._is_friendly(piece, enemy):
            return
        
        # Check if enemy is a king
        if enemy == KING:
            self._check_king_capture(nr, nc)
            return
        
        # Regular piece: check if surrounded on opposite side
        nr2, nc2 = nr + dr, nc + dc
        
        if not (0 <= nr2 < 7 and 0 <= nc2 < 7):
            return
        
        # Can be captured against friendly piece, throne, or corner
        opposite = self.board[nr2, nc2]
        is_hostile_square = (nr2, nc2) == self.throne or (nr2, nc2) in self.corner_set
        
        if self._is_friendly(piece, opposite) or is_hostile_square:
            self.board[nr, nc] = EMPTY
    
    def _check_king_capture(self, r: int, c: int):
        """Check if the king at (r, c) is captured."""
        # King must be surrounded on all 4 sides
        adjacent_hostile = 0
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            
            if not (0 <= nr < 7 and 0 <= nc < 7):
                adjacent_hostile += 1  # Edge counts as hostile
                continue
            
            piece = self.board[nr, nc]
            is_hostile_square = (nr, nc) == self.throne or (nr, nc) in self.corner_set
            
            if piece == ATTACKER or is_hostile_square:
                adjacent_hostile += 1
        
        if adjacent_hostile == 4:
            self.board[r, c] = EMPTY
            self.game_over = True
            self.winner = ATTACKER_PLAYER
    
    def _is_friendly(self, piece1: int, piece2: int) -> bool:
        """Check if two pieces are on the same team."""
        if piece1 == EMPTY or piece2 == EMPTY:
            return False
        
        if piece1 == ATTACKER:
            return piece2 == ATTACKER
        else:  # piece1 is DEFENDER or KING
            return piece2 in [DEFENDER, KING]
    
    def _check_game_over(self):
        """Check if the game is over."""
        # Check if king escaped to corner
        for corner in self.corners:
            if self.board[corner] == KING:
                self.game_over = True
                self.winner = DEFENDER_PLAYER
                return
        
        # Check if king is captured (already handled in _check_king_capture)
        if KING not in self.board:
            self.game_over = True
            self.winner = ATTACKER_PLAYER
            return
        
        # Check if current player has no legal moves (stalemate = loss)
        # Note: Three-fold repetition is now handled by filtering illegal moves,
        # so a player with only repetition moves will have no legal moves
        if not self.get_legal_moves():
            self.game_over = True
            self.winner = 1 - self.current_player
            return
    
    def clone(self):
        """Create a deep copy of the game state."""
        new_game = Brandubh.__new__(Brandubh)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.throne = self.throne
        new_game.corners = self.corners
        new_game.corner_set = self.corner_set
        new_game.position_history = self.position_history.copy()
        new_game.move_count = self.move_count
        new_game.king_has_left_throne = self.king_has_left_throne
        return new_game
    
    def __str__(self) -> str:
        """Pretty print the board."""
        symbols = {
            EMPTY: '·',
            ATTACKER: '○',
            DEFENDER: '●',
            KING: '♔'
        }
        
        lines = []
        lines.append("  " + " ".join(str(i) for i in range(7)))
        lines.append("  " + "─" * 13)
        
        for r in range(7):
            row_str = f"{r}│"
            for c in range(7):
                # Mark special squares
                if (r, c) in self.corner_set:
                    if self.board[r, c] == EMPTY:
                        row_str += "X "
                    else:
                        row_str += symbols[self.board[r, c]] + " "
                elif (r, c) == self.throne:
                    if self.board[r, c] == EMPTY:
                        row_str += "⊕ "
                    else:
                        row_str += symbols[self.board[r, c]] + " "
                else:
                    row_str += symbols[self.board[r, c]] + " "
            lines.append(row_str)
        
        player_name = "Attackers (○)" if self.current_player == ATTACKER_PLAYER else "Defenders (●/♔)"
        lines.append(f"\nCurrent player: {player_name}")
        
        if self.game_over:
            winner_name = "Attackers" if self.winner == ATTACKER_PLAYER else "Defenders"
            lines.append(f"Game Over! Winner: {winner_name}")
        
        return "\n".join(lines)
