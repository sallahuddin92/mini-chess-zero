"""
Mini-Chess Zero Web API
=======================
FastAPI backend for the chess web app.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment import MiniChessEnv
from src.agent import DQNAgent

app = FastAPI(
    title="Mini-Chess Zero",
    description="Play against an AI trained with Asymmetric TD Learning",
    version="1.0.0"
)

# Global game state
games = {}


class GameState(BaseModel):
    board: List[int]
    turn: int  # 1 = White, -1 = Black
    legal_moves: List[str]
    game_over: bool
    winner: Optional[str] = None
    last_move: Optional[str] = None
    ai_thinking: bool = False


class MoveRequest(BaseModel):
    game_id: str
    move: str  # e.g., "a2a3"


class NewGameRequest(BaseModel):
    difficulty: str = "intermediate"  # beginner, intermediate, advanced, expert
    player_color: str = "white"  # white or black


# Helper functions
COLS = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
INV_COLS = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}

def index_to_notation(idx: int) -> str:
    row = idx // 5
    col = idx % 5
    return f"{INV_COLS[col]}{row + 1}"

def notation_to_indices(move: str) -> tuple:
    try:
        c1, r1, c2, r2 = move[0], int(move[1]), move[2], int(move[3])
        start = (r1 - 1) * 5 + COLS[c1]
        end = (r2 - 1) * 5 + COLS[c2]
        return start, end
    except:
        return None, None

def get_legal_moves_notation(env: MiniChessEnv) -> List[str]:
    legal = env.game.get_legal_moves()
    return [f"{index_to_notation(s)}{index_to_notation(e)}" for s, e in legal]


class GameSession:
    DIFFICULTY_CONFIG = {
        "beginner": {"epsilon": 0.5, "noise": 0.3},
        "intermediate": {"epsilon": 0.2, "noise": 0.1},
        "advanced": {"epsilon": 0.05, "noise": 0.02},
        "expert": {"epsilon": 0.0, "noise": 0.0}
    }
    
    def __init__(self, difficulty: str = "intermediate", player_color: str = "white"):
        self.env = MiniChessEnv()
        self.agent = DQNAgent(self.env.state_size, self.env.action_size)
        self.difficulty = difficulty
        self.player_color = player_color
        self.game_over = False
        self.winner = None
        self.last_move = None
        
        # Load model
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "final_model.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.agent.device)
            if isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
                self.agent.policy_net.load_state_dict(checkpoint['policy_net'])
            self.agent.epsilon = 0.0
        
        self.state = self.env.reset()
    
    def get_state(self) -> GameState:
        return GameState(
            board=list(self.env.game.board),
            turn=self.env.game.turn,
            legal_moves=get_legal_moves_notation(self.env),
            game_over=self.game_over,
            winner=self.winner,
            last_move=self.last_move
        )
    
    def is_player_turn(self) -> bool:
        if self.player_color == "white":
            return self.env.game.turn == 1
        return self.env.game.turn == -1
    
    def make_move(self, move: str) -> GameState:
        start, end = notation_to_indices(move)
        if start is None:
            raise ValueError("Invalid move format")
        
        legal = self.env.game.get_legal_moves()
        if (start, end) not in legal:
            raise ValueError("Illegal move")
        
        action = start * 25 + end
        self.state, reward, done, _ = self.env.step(action)
        self.last_move = move
        
        if done:
            self.game_over = True
            self.winner = "white" if self.env.game.turn == -1 else "black"
        
        return self.get_state()
    
    def ai_move(self) -> GameState:
        if self.game_over:
            return self.get_state()
        
        config = self.DIFFICULTY_CONFIG[self.difficulty]
        mask = self.env.get_action_mask()
        
        if not np.any(mask):
            self.game_over = True
            self.winner = "white" if self.env.game.turn == -1 else "black"
            return self.get_state()
        
        # Add randomness for difficulty
        if np.random.random() < config["epsilon"]:
            legal_indices = np.where(mask)[0]
            action = np.random.choice(legal_indices)
        else:
            state_t = torch.FloatTensor(self.state).unsqueeze(0).to(self.agent.device)
            with torch.no_grad():
                q_values = self.agent.policy_net(state_t).cpu().numpy()[0]
                # Add noise
                q_values += np.random.normal(0, config["noise"], q_values.shape)
                q_values[~mask] = float('-inf')
                action = np.argmax(q_values)
        
        start = action // 25
        end = action % 25
        self.last_move = f"{index_to_notation(start)}{index_to_notation(end)}"
        
        self.state, reward, done, _ = self.env.step(action)
        
        if done:
            self.game_over = True
            self.winner = "white" if self.env.game.turn == -1 else "black"
        
        return self.get_state()


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def home():
    return FileResponse("templates/index.html")


@app.post("/api/new_game")
async def new_game(request: NewGameRequest):
    import uuid
    game_id = str(uuid.uuid4())[:8]
    games[game_id] = GameSession(
        difficulty=request.difficulty,
        player_color=request.player_color
    )
    
    state = games[game_id].get_state()
    
    # If AI plays first (player is black)
    if request.player_color == "black":
        state = games[game_id].ai_move()
    
    return {"game_id": game_id, **state.dict()}


@app.post("/api/move")
async def make_move(request: MoveRequest):
    if request.game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = games[request.game_id]
    
    if game.game_over:
        raise HTTPException(status_code=400, detail="Game is over")
    
    if not game.is_player_turn():
        raise HTTPException(status_code=400, detail="Not your turn")
    
    try:
        state = game.make_move(request.move)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # AI responds
    if not state.game_over:
        state = game.ai_move()
    
    return state.dict()


@app.get("/api/state/{game_id}")
async def get_game_state(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    return games[game_id].get_state().dict()


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": True}


# Mount static files (relative to webapp directory)
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
