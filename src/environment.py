import numpy as np
import copy

class MiniChessBoard:
    def __init__(self):
        # 5x5 Board (0-24)
        # 1:P, 2:N, 3:B, 4:R, 5:Q, 6:K
        # Positive = White, Negative = Black
        self.board = np.zeros(25, dtype=int)
        self.reset()
        
    def reset(self):
        # Gardner Mini-Chess Setup
        # White (Rows 0-1)
        self.board[0:5] = [4, 2, 3, 5, 6] # R, N, B, Q, K
        self.board[5:10] = [1, 1, 1, 1, 1] # Pawns
        # Empty (Row 2)
        self.board[10:15] = 0
        # Black (Rows 3-4) - Mirrored
        self.board[15:20] = [-1, -1, -1, -1, -1]
        self.board[20:25] = [-4, -2, -3, -5, -6] 
        self.turn = 1 # 1=White, -1=Black
        
    def get_legal_moves(self):
        """Generates all legal (start, end) tuples."""
        moves = []
        for idx in range(25):
            piece = self.board[idx]
            if piece * self.turn <= 0: continue # Not my piece
            
            p_type = abs(piece)
            row, col = divmod(idx, 5)
            
            # --- MOVEMENT LOGIC ---
            targets = []
            
            # 1. PAWN
            if p_type == 1:
                direction = 1 if self.turn == 1 else -1
                fwd = idx + (direction * 5)
                if 0 <= fwd < 25 and self.board[fwd] == 0:
                    moves.append((idx, fwd))
                for diag in [-1, 1]:
                    if (col == 0 and diag == -1) or (col == 4 and diag == 1): continue
                    tgt = fwd + diag
                    if 0 <= tgt < 25 and self.board[tgt] * self.turn < 0:
                        moves.append((idx, tgt))
                continue 

            # 2. KNIGHT
            elif p_type == 2:
                offsets = [-11, -9, -7, -3, 3, 7, 9, 11]
                for off in offsets:
                    t = idx + off
                    tr, tc = divmod(t, 5)
                    if 0 <= t < 25 and abs(tr-row) + abs(tc-col) == 3:
                        targets.append(t)

            # 3. KING
            elif p_type == 6:
                offsets = [-6, -5, -4, -1, 1, 4, 5, 6]
                for off in offsets:
                    t = idx + off
                    tr, tc = divmod(t, 5)
                    if 0 <= t < 25 and abs(tr-row) <= 1 and abs(tc-col) <= 1:
                        targets.append(t)

            # 4. SLIDERS (R, B, Q)
            elif p_type in [3, 4, 5]:
                dirs = []
                if p_type in [4, 5]: dirs += [-5, 5, -1, 1] # R/Q
                if p_type in [3, 5]: dirs += [-6, -4, 4, 6] # B/Q
                
                for d in dirs:
                    curr = idx
                    while True:
                        c_row, c_col = divmod(curr, 5)
                        if (d in [-1, -6, 4] and c_col == 0): break
                        if (d in [1, -4, 6] and c_col == 4): break
                        
                        curr += d
                        if not (0 <= curr < 25): break
                        
                        tp = self.board[curr]
                        if tp == 0:
                            moves.append((idx, curr))
                        else:
                            if tp * self.turn < 0: moves.append((idx, curr))
                            break
            
            for t in targets:
                if self.board[t] * self.turn <= 0:
                    moves.append((idx, t))
                    
        return moves

    def push(self, move):
        start, end = move
        captured = self.board[end]
        self.board[end] = self.board[start]
        self.board[start] = 0
        
        # Auto-Queen
        if abs(self.board[end]) == 1:
            row = end // 5
            if (self.turn == 1 and row == 4) or (self.turn == -1 and row == 0):
                self.board[end] = 5 * self.turn

        self.turn *= -1
        return captured

class MiniChessEnv:
    def __init__(self):
        self.game = MiniChessBoard()
        self.state_size = 25
        self.action_size = 625 
    
    def reset(self):
        self.game.reset()
        return self.get_canonical_state()

    def get_action_mask(self):
        """
        RESEARCH GRADE: Action Masking
        Returns boolean array (625,) where True = Legal Move.
        """
        legal_moves = self.game.get_legal_moves()
        mask = np.zeros(self.action_size, dtype=bool)
        for start, end in legal_moves:
            idx = start * 25 + end
            mask[idx] = True
        return mask

    def step(self, action_idx):
        start = action_idx // 25
        end = action_idx % 25
        move = (start, end)
        
        # Verify legality (Masking should prevent this, but safety first)
        legal_moves = self.game.get_legal_moves()
        if move not in legal_moves:
            # Illegal move gets small penalty, not huge -10
            return self.get_canonical_state(), -0.5, True, {'material': 0, 'illegal': True}
        
        captured = self.game.push(move)
        
        # ===========================================
        # RESEARCH-GRADE REWARD SHAPING
        # All rewards bounded in [-1, 1] range
        # ===========================================
        
        # Piece values (normalized to ~0.1-1.0 scale)
        PIECE_VALUES = {
            1: 0.10,  # Pawn
            2: 0.30,  # Knight  
            3: 0.30,  # Bishop
            4: 0.50,  # Rook
            5: 0.90,  # Queen
            6: 1.00,  # King (terminal)
        }
        
        reward = -0.005  # Small step penalty (encourages faster wins)
        done = False
        
        if abs(captured) == 6:  # King Capture = Win
            reward = 1.0
            done = True
        elif captured != 0:  # Material capture
            reward = PIECE_VALUES.get(abs(captured), 0.1)
            
        return self.get_canonical_state(), reward, done, {'material': captured}


    def get_canonical_state(self):
        state = self.game.board.copy()
        if self.game.turn == -1:
            state = state[::-1] * -1
        return state / 6.0