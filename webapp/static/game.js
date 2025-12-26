/**
 * Mini-Chess Zero - Game Logic
 */

// Piece symbols
const PIECES = {
    1: '♙', 2: '♘', 3: '♗', 4: '♖', 5: '♕', 6: '♔',   // White
    '-1': '♟', '-2': '♞', '-3': '♝', '-4': '♜', '-5': '♛', '-6': '♚'  // Black
};

// Board columns
const COLS = ['a', 'b', 'c', 'd', 'e'];

// Game state
let gameId = null;
let selectedSquare = null;
let legalMoves = [];
let playerColor = 'white';
let difficulty = 'intermediate';
let moveHistory = [];

// DOM Elements
const startScreen = document.getElementById('start-screen');
const gameScreen = document.getElementById('game-screen');
const board = document.getElementById('chess-board');
const turnIndicator = document.getElementById('turn-indicator');
const movesList = document.getElementById('moves-list');
const gameOverModal = document.getElementById('game-over-modal');
const resultText = document.getElementById('result-text');

// Initialize button groups
document.querySelectorAll('.button-group').forEach(group => {
    group.querySelectorAll('button').forEach(btn => {
        btn.addEventListener('click', () => {
            group.querySelectorAll('button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            if (group.id === 'difficulty-select') {
                difficulty = btn.dataset.value;
            } else if (group.id === 'color-select') {
                playerColor = btn.dataset.value;
            }
        });
    });
});

// Start button
document.getElementById('start-btn').addEventListener('click', startNewGame);
document.getElementById('new-game-btn').addEventListener('click', () => {
    startScreen.classList.remove('hidden');
    gameScreen.classList.add('hidden');
});
document.getElementById('play-again-btn').addEventListener('click', () => {
    gameOverModal.classList.add('hidden');
    startNewGame();
});

// Start new game
async function startNewGame() {
    try {
        const response = await fetch('/api/new_game', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                difficulty: difficulty,
                player_color: playerColor
            })
        });
        
        const data = await response.json();
        gameId = data.game_id;
        legalMoves = data.legal_moves;
        moveHistory = [];
        
        startScreen.classList.add('hidden');
        gameScreen.classList.remove('hidden');
        
        renderBoard(data.board, data.last_move);
        updateTurnIndicator(data.turn);
        
        if (data.last_move) {
            addMoveToHistory(data.last_move, 'ai');
        }
        
    } catch (error) {
        console.error('Failed to start game:', error);
        alert('Failed to start game. Is the server running?');
    }
}

// Render the board
function renderBoard(boardData, lastMove = null) {
    board.innerHTML = '';
    
    // Determine if we need to flip the board
    const flip = playerColor === 'black';
    
    for (let displayRow = 4; displayRow >= 0; displayRow--) {
        for (let displayCol = 0; displayCol < 5; displayCol++) {
            const row = flip ? 4 - displayRow : displayRow;
            const col = flip ? 4 - displayCol : displayCol;
            const idx = row * 5 + col;
            
            const square = document.createElement('div');
            square.className = 'square ' + ((row + col) % 2 === 0 ? 'dark' : 'light');
            square.dataset.idx = idx;
            
            const piece = boardData[idx];
            if (piece !== 0) {
                square.textContent = PIECES[piece] || PIECES[String(piece)];
            }
            
            // Highlight last move
            if (lastMove) {
                const notation = indexToNotation(idx);
                if (lastMove.includes(notation)) {
                    square.classList.add('last-move');
                }
            }
            
            square.addEventListener('click', () => handleSquareClick(idx, boardData));
            board.appendChild(square);
        }
    }
}

// Convert index to notation
function indexToNotation(idx) {
    const row = Math.floor(idx / 5);
    const col = idx % 5;
    return COLS[col] + (row + 1);
}

// Handle square click
function handleSquareClick(idx, boardData) {
    const notation = indexToNotation(idx);
    
    // If already selected, try to move
    if (selectedSquare !== null) {
        const from = indexToNotation(selectedSquare);
        const move = from + notation;
        
        if (legalMoves.includes(move)) {
            makeMove(move);
        } else {
            // Select new piece if clicking own piece
            if (boardData[idx] !== 0) {
                selectSquare(idx, boardData);
            } else {
                clearSelection();
            }
        }
    } else {
        // Select piece
        if (boardData[idx] !== 0) {
            selectSquare(idx, boardData);
        }
    }
}

// Select a square
function selectSquare(idx, boardData) {
    clearSelection();
    selectedSquare = idx;
    
    const from = indexToNotation(idx);
    const squares = board.querySelectorAll('.square');
    
    squares.forEach(sq => {
        if (parseInt(sq.dataset.idx) === idx) {
            sq.classList.add('selected');
        }
        
        // Show legal moves
        const to = indexToNotation(parseInt(sq.dataset.idx));
        if (legalMoves.includes(from + to)) {
            sq.classList.add('legal-move');
        }
    });
}

// Clear selection
function clearSelection() {
    selectedSquare = null;
    board.querySelectorAll('.square').forEach(sq => {
        sq.classList.remove('selected', 'legal-move');
    });
}

// Make a move
async function makeMove(move) {
    turnIndicator.textContent = 'AI Thinking...';
    turnIndicator.className = 'turn-indicator ai-turn';
    
    try {
        const response = await fetch('/api/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                game_id: gameId,
                move: move
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            alert(error.detail || 'Invalid move');
            return;
        }
        
        const data = await response.json();
        legalMoves = data.legal_moves;
        
        addMoveToHistory(move, 'player');
        
        if (data.last_move && data.last_move !== move) {
            addMoveToHistory(data.last_move, 'ai');
        }
        
        renderBoard(data.board, data.last_move);
        clearSelection();
        
        if (data.game_over) {
            showGameOver(data.winner);
        } else {
            updateTurnIndicator(data.turn);
        }
        
    } catch (error) {
        console.error('Move failed:', error);
        alert('Failed to make move');
    }
}

// Update turn indicator
function updateTurnIndicator(turn) {
    const isPlayerTurn = (playerColor === 'white' && turn === 1) || 
                        (playerColor === 'black' && turn === -1);
    
    if (isPlayerTurn) {
        turnIndicator.textContent = 'Your Turn';
        turnIndicator.className = 'turn-indicator your-turn';
    } else {
        turnIndicator.textContent = 'AI Turn';
        turnIndicator.className = 'turn-indicator ai-turn';
    }
}

// Add move to history
function addMoveToHistory(move, player) {
    moveHistory.push({ move, player });
    
    const entry = document.createElement('div');
    entry.className = 'move-entry';
    
    const num = document.createElement('span');
    num.className = 'move-number';
    num.textContent = Math.ceil(moveHistory.length / 2) + '.';
    
    const moveSpan = document.createElement('span');
    moveSpan.className = player === 'player' ? 'move-white' : 'move-black';
    moveSpan.textContent = move;
    
    entry.appendChild(num);
    entry.appendChild(moveSpan);
    movesList.appendChild(entry);
    movesList.scrollTop = movesList.scrollHeight;
}

// Show game over
function showGameOver(winner) {
    const playerWon = winner === playerColor;
    
    if (playerWon) {
        resultText.textContent = '🎉 You Win!';
        resultText.style.color = 'var(--accent-primary)';
    } else {
        resultText.textContent = '💀 AI Wins';
        resultText.style.color = 'var(--danger)';
    }
    
    gameOverModal.classList.remove('hidden');
}
