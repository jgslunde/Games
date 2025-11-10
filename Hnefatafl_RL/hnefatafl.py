"""
Hnefatafl (11x11 Tafl) game implementation.
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
- King is captured like any other piece (between two hostile tiles)
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


class Hnefatafl:
    def __init__(self, 
                 king_capture_pieces: int = 2,
                 king_can_capture: bool = True,
                 throne_is_hostile: bool = False,
                 throne_enabled: bool = True):
        """
        Initialize Hnefatafl game with tunable rules.
        
        Args:
            king_capture_pieces: Number of pieces required to capture the king (2, 3, or 4).
                - 2: King captured between two attackers (standard custodian capture)
                - 3: King must be surrounded on 3 sides
                - 4: King must be surrounded on all 4 sides
            king_can_capture: Whether the king can participate in capturing enemy pieces.
            throne_is_hostile: Whether the throne (center square) acts as a hostile square
                             for captures (like corners do). Only applies if throne_enabled=True.
            throne_enabled: Whether the throne exists and restricts movement (only king can 
                          move to throne). If False, center square acts like any other square.
        """
        self.board = np.zeros((11, 11), dtype=np.int8)
        self.current_player = ATTACKER_PLAYER  # Attackers start
        self.game_over = False
        self.winner = None
        
        # Tunable rules
        self.king_capture_pieces = king_capture_pieces
        self.king_can_capture = king_can_capture
        self.throne_is_hostile = throne_is_hostile
        self.throne_enabled = throne_enabled
        
        # Validate rules
        if king_capture_pieces not in [2, 3, 4]:
            raise ValueError("king_capture_pieces must be 2, 3, or 4")
        
        # Special squares
        self.throne = (5, 5)
        self.corners = [(0, 0), (0, 10), (10, 0), (10, 10)]
        self.corner_set = set(self.corners)
        
        # Position history for repetition detection (3-fold repetition = illegal move)
        self.position_history = []
        self.move_count = 0
        
        self._setup_board()
        self._record_position()
    
    def _setup_board(self):
        """Initialize the starting position for Hnefatafl."""
        # King at center
        self.board[5, 5] = KING
        
        # Defenders around king (diamond formation - 12 defenders)
        defenders = [
            # Inner diamond (4 squares adjacent to king)
            (4, 5), (5, 4), (5, 6), (6, 5),
            # Outer diamond points (4 squares)
            (3, 5), (5, 3), (5, 7), (7, 5),
            # Diagonal corners of diamond (4 squares)
            (4, 4), (4, 6), (6, 4), (6, 6),
        ]
        for r, c in defenders:
            self.board[r, c] = DEFENDER
        
        # Attackers on edges (standard Hnefatafl layout - 24 attackers)
        attackers = [
            # Top edge
            (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
            (1, 5),
            # Right edge
            (3, 10), (4, 10), (5, 10), (6, 10), (7, 10),
            (5, 9),
            # Bottom edge
            (10, 3), (10, 4), (10, 5), (10, 6), (10, 7),
            (9, 5),
            # Left edge
            (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
            (5, 1)
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
        """Check if the current position has occurred 3 times (illegal to cause third repetition)."""
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
        state = np.zeros((4, 11, 11), dtype=np.float32)
        state[0] = (self.board == ATTACKER).astype(np.float32)
        state[1] = (self.board == DEFENDER).astype(np.float32)
        state[2] = (self.board == KING).astype(np.float32)
        state[3] = np.full((11, 11), self.current_player, dtype=np.float32)
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
        
        for r in range(11):
            for c in range(11):
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
            while 0 <= nr < 11 and 0 <= nc < 11:
                # Can't move to occupied square
                if self.board[nr, nc] != EMPTY:
                    break
                
                # Only king can move to corners (always restricted)
                # Only king can move to throne (if throne is enabled)
                is_king = self.board[r, c] == KING
                is_corner = (nr, nc) in self.corner_set
                is_throne = (nr, nc) == self.throne
                
                # Block non-king from corners (always) - cannot move through corners
                if is_corner and not is_king:
                    break
                
                # Block non-king from landing on throne (only if throne is enabled)
                # But allow moving THROUGH the throne (continue to next square)
                if is_throne and self.throne_enabled and not is_king:
                    nr += dr
                    nc += dc
                    continue  # Skip this square but keep checking beyond it
                
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
        if not (0 <= nr < 11 and 0 <= nc < 11):
            return
        
        enemy = self.board[nr, nc]
        if enemy == EMPTY or self._is_friendly(piece, enemy):
            return
        
        target_is_king = enemy == KING
        
        # Handle king capture based on rules
        if target_is_king:
            # For 2-piece capture, check if we complete a sandwich in THIS direction
            if self.king_capture_pieces == 2:
                nr2, nc2 = nr + dr, nc + dc
                if 0 <= nr2 < 11 and 0 <= nc2 < 11 and (self.board[nr2, nc2] == ATTACKER or self._is_hostile_square(nr2, nc2, KING)):
                    # Complete sandwich in this direction
                    self.board[nr, nc] = EMPTY
                    self.game_over = True
                    self.winner = ATTACKER_PLAYER
            else:
                # For 3 or 4 piece captures, check all sides
                if self._is_king_captured(nr, nc):
                    self.board[nr, nc] = EMPTY
                    self.game_over = True
                    self.winner = ATTACKER_PLAYER
        else:
            # Regular piece capture - check opposite side
            nr2, nc2 = nr + dr, nc + dc
            
            if not (0 <= nr2 < 11 and 0 <= nc2 < 11):
                return
            
            opposite = self.board[nr2, nc2]
            is_hostile_square = self._is_hostile_square(nr2, nc2, enemy)
            
            if self._is_friendly(piece, opposite) or is_hostile_square:
                self.board[nr, nc] = EMPTY
    
    def _is_king_captured(self, king_r: int, king_c: int) -> bool:
        """
        Check if king at (king_r, king_c) is captured based on current rules.
        
        Returns:
            True if king is captured, False otherwise
        """
        if self.king_capture_pieces == 2:
            # Standard custodian capture - need attackers/hostile squares on opposite sides
            # Check horizontal
            left_hostile = (king_c > 0 and (self.board[king_r, king_c - 1] == ATTACKER or 
                           self._is_hostile_square(king_r, king_c - 1, KING)))
            right_hostile = (king_c < 10 and (self.board[king_r, king_c + 1] == ATTACKER or 
                            self._is_hostile_square(king_r, king_c + 1, KING)))
            if left_hostile and right_hostile:
                return True
            
            # Check vertical
            top_hostile = (king_r > 0 and (self.board[king_r - 1, king_c] == ATTACKER or 
                          self._is_hostile_square(king_r - 1, king_c, KING)))
            bottom_hostile = (king_r < 10 and (self.board[king_r + 1, king_c] == ATTACKER or 
                             self._is_hostile_square(king_r + 1, king_c, KING)))
            if top_hostile and bottom_hostile:
                return True
            return False
            
        elif self.king_capture_pieces == 3:
            # Need attackers/hostile squares on 3 out of 4 sides
            hostile_count = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = king_r + dr, king_c + dc
                if 0 <= nr < 11 and 0 <= nc < 11:
                    if self.board[nr, nc] == ATTACKER or self._is_hostile_square(nr, nc, KING):
                        hostile_count += 1
            return hostile_count >= 3
            
        elif self.king_capture_pieces == 4:
            # Need attackers/hostile squares on all 4 sides
            # Board edges also count as hostile in 4-piece mode
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = king_r + dr, king_c + dc
                # Check if out of bounds (edge of board counts as hostile in 4-piece mode)
                if not (0 <= nr < 11 and 0 <= nc < 11):
                    continue  # Edge is hostile, this side is satisfied
                # Check if square has attacker or is a hostile square
                if self.board[nr, nc] != ATTACKER and not self._is_hostile_square(nr, nc, KING):
                    return False
            return True
        
        return False
    
    def _is_hostile_square(self, r: int, c: int, target_piece: int) -> bool:
        """
        Check if a square is hostile for capturing purposes.
        target_piece: The piece type being captured (ATTACKER, DEFENDER, or KING).
        """
        # Corners are always hostile
        if (r, c) in self.corner_set:
            return True
        
        # Throne: depends on whether it's occupied and what piece is being captured
        if self.throne_enabled and (r, c) == self.throne:
            # If throne is occupied, it's not hostile (can't squeeze against a piece)
            if self.board[r, c] != EMPTY:
                return False
            # Empty throne: hostile to attackers, but only hostile to defenders/king if throne_is_hostile
            if target_piece == ATTACKER:
                return True  # Always hostile to attackers
            else:
                # For defenders and king, only hostile if throne_is_hostile rule is set
                return self.throne_is_hostile
        
        return False
    
    def _is_friendly(self, piece1: int, piece2: int) -> bool:
        """Check if two pieces are on the same team."""
        if piece1 == EMPTY or piece2 == EMPTY:
            return False
        
        if piece1 == ATTACKER:
            return piece2 == ATTACKER
        else:  # piece1 is DEFENDER or KING
            # If king cannot capture, king doesn't participate in captures
            if not self.king_can_capture:
                # King is only friendly with itself
                if piece1 == KING:
                    return piece2 == KING
                # Defenders don't treat king as friendly for capture purposes
                if piece2 == KING:
                    return False
            # Otherwise defenders and king are friendly
            return piece2 in [DEFENDER, KING]
    
    def _is_king_encircled(self) -> bool:
        """
        Check if the king is encircled by attackers.
        
        Uses flood-fill algorithm from the king's position. If the king's group
        (king + any connected defenders/empty squares) cannot reach the edge of
        the board without crossing attacker pieces, the king is encircled.
        
        This is efficient O(n) where n is the number of squares checked.
        """
        # Find the king's position
        king_pos = None
        for r in range(11):
            for c in range(11):
                if self.board[r, c] == KING:
                    king_pos = (r, c)
                    break
            if king_pos:
                break
        
        if not king_pos:
            return False  # No king on board (already captured)
        
        # Flood fill from king's position
        # Can move through empty squares, defenders, and the king
        # Cannot move through attackers
        visited = set()
        stack = [king_pos]
        visited.add(king_pos)
        
        while stack:
            r, c = stack.pop()
            
            # If we reached an edge, king is NOT encircled
            if r == 0 or r == 10 or c == 0 or c == 10:
                return False
            
            # Check all 4 adjacent squares
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                
                # Skip if out of bounds
                if not (0 <= nr < 11 and 0 <= nc < 11):
                    continue
                
                # Skip if already visited
                if (nr, nc) in visited:
                    continue
                
                # Can only move through non-attacker squares
                if self.board[nr, nc] != ATTACKER:
                    visited.add((nr, nc))
                    stack.append((nr, nc))
        
        # If we exhausted the flood fill without reaching an edge, king is encircled
        return True
    
    def _check_game_over(self):
        """Check if the game is over."""
        # Check if king escaped to corner
        for corner in self.corners:
            if self.board[corner] == KING:
                self.game_over = True
                self.winner = DEFENDER_PLAYER
                return
        
        # Check if king was captured during the previous move resolution
        if KING not in self.board:
            self.game_over = True
            self.winner = ATTACKER_PLAYER
            return
        
        # Check for encirclement (Hnefatafl-specific rule)
        # Attackers win if they form a continuous chain surrounding the king
        if self._is_king_encircled():
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
        new_game = Hnefatafl.__new__(Hnefatafl)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.throne = self.throne
        new_game.corners = self.corners
        new_game.corner_set = self.corner_set
        new_game.position_history = self.position_history.copy()
        new_game.move_count = self.move_count
        # Copy rule parameters
        new_game.king_capture_pieces = self.king_capture_pieces
        new_game.king_can_capture = self.king_can_capture
        new_game.throne_is_hostile = self.throne_is_hostile
        new_game.throne_enabled = self.throne_enabled
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
        lines.append("   " + " ".join(f"{i:1}" for i in range(10)) + " A")
        lines.append("   " + "─" * 21)
        
        for r in range(11):
            if r < 10:
                row_str = f"{r} │"
            else:
                row_str = f"A│"
            for c in range(11):
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
