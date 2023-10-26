const HnefataflGame = {
    name: 'hnefatafl',

    setup: () => ({
        cells: 
        [
            ['0', '0', '0', 'A', '0', '0', '0'],
            ['0', '0', '0', 'A', '0', '0', '0'],
            ['0', '0', '0', 'D', '0', '0', '0'],
            ['A', 'A', 'D', 'K', 'D', 'A', 'A'],
            ['0', '0', '0', 'D', '0', '0', '0'],
            ['0', '0', '0', 'A', '0', '0', '0'],
            ['0', '0', '0', 'A', '0', '0', '0']
            ],
        selected: null,
        validMoves: [],
        score: 0,
    }),

      moves: {
        selectPiece: ({G, ctx}, x, y) => {
            const piece = G.cells[y][x];
            if (ctx.currentPlayer === '0' && piece !== 'A') return;
            if (ctx.currentPlayer === '1' && (piece !== 'D' && piece !== 'K')) return;
        
            if (piece !== '0') {
                G.selected = { x, y };
                G.validMoves = getValidMoves(G.cells, x, y);
            }
        },

        deselectPiece: ({G}) => {
            G.selected = null;
            G.validMoves = [];
        },
            
        movePiece: ({G, ctx, events}, x, y) => {
            if (G.selected && G.validMoves.some(move => move.x === x && move.y === y)) {
                const fromX = G.selected.x;
                const fromY = G.selected.y;
    
                // Swap pieces
                G.cells[y][x] = G.cells[fromY][fromX];
                G.cells[fromY][fromX] = '0';
    
                // Clear selection and valid moves
                G.selected = null;
                G.validMoves = [];
                
                checkCapture(G, x, y);

                G.score = calculateScore(G.cells);

                // Check for win conditions
                const winner = checkWinCondition(G);
                if (winner) {
                    events.endGame({ winner });
                }
                events.endTurn();  // This will end the turn and switch to the next player.
            }
        },

        makeRandomMove: ({G, ctx, events}) => {
            let moves = [];    
            // Loop through all cells to find possible moves
            for (let y = 0; y < 7; y++) {
                for (let x = 0; x < 7; x++) {
                    const cell = G.cells[y][x];
                    if ((cell === 'A' && ctx.currentPlayer === '0') || 
                        (cell === 'D' && ctx.currentPlayer === '1') ||
                        (cell === 'K' && ctx.currentPlayer === '1')) {
                        const validMoves = getValidMoves(G.cells, x, y); 
                        for (const move of validMoves) {
                        // moves.push({ move: 'movePiece', args: [x, y, move.x, move.y] });
                        moves.push([x, y, move.x, move.y]);
                        }
                    }
                }
            }
            // if (moves.length > 0) {
            const randomMove = moves[Math.floor(Math.random() * moves.length)];        
            // }
            let fromX = randomMove[0];
            let fromY = randomMove[1];
            let x = randomMove[2];
            let y = randomMove[3];
            G.selected = { x, y };
            // Swap pieces
            G.cells[y][x] = G.cells[fromY][fromX];
            G.cells[fromY][fromX] = '0';

            // Clear selection and valid moves
            G.selected = null;
            G.validMoves = [];
            
            checkCapture(G, x, y);

            G.score = calculateScore(G.cells);

            // Check for win conditions
            const winner = checkWinCondition(G);
            if (winner) {
                events.endGame({ winner });
            }
            events.endTurn();  // This will end the turn and switch to the next player.
        }
    },

    // Add any other game-specific configurations, e.g., turn order, endgame conditions, etc.
};


function checkCapture(G, x, y) {
    const currentPiece = G.cells[y][x];

    const directions = [
        { x: 1, y: 0 },  // right
        { x: -1, y: 0 }, // left
        { x: 0, y: 1 },  // down
        { x: 0, y: -1 }  // up
    ];

    for (const dir of directions) {
        const enemyX = x + dir.x;
        const enemyY = y + dir.y;
        
        // Check if the cell exists and contains an enemy piece
        if (isInBounds(enemyX, enemyY) && isEnemy(G.cells[enemyY][enemyX], currentPiece)) {
            const allyX = enemyX + dir.x;
            const allyY = enemyY + dir.y;
            
            // If the piece opposite the enemy is an ally, a corner, or the cell itself is a corner, capture the enemy
            if (isInBounds(allyX, allyY) && ((isAlly(G.cells[allyY][allyX], currentPiece) || isCorner(allyX, allyY)) || isCorner(enemyX, enemyY))){
                G.cells[enemyY][enemyX] = '0';  // Capture enemy by setting its position to '0'
            }
        }
    }
}

function isEnemy(piece, currentPiece) {
    return ((piece === 'A') && (currentPiece !== 'A')) || ((piece !== 'A') && (piece !== '0') && (currentPiece === 'A'));
}

function isAlly(piece, currentPiece) {
    return !isEnemy(piece, currentPiece) && piece !== '0';
}

function isInBounds(x, y) {
    return x >= 0 && x < 7 && y >= 0 && y < 7;
}

function isCorner(x, y) {
    return ((x === 0) && (y === 0)) || ((x === 0) && (y === 6)) || ((x === 6) && (y === 0)) || ((x === 6) && (y === 6));
}



function getValidMoves(cells, x, y) {
    const directions = [
        { x: 1, y: 0 },   // right
        { x: -1, y: 0 },  // left
        { x: 0, y: 1 },   // down
        { x: 0, y: -1 },  // up
    ];

    let validMoves = [];

    for (const dir of directions) {
        let newX = x + dir.x;
        let newY = y + dir.y;

        while (newX >= 0 && newX < 7 && newY >= 0 && newY < 7 && cells[newY][newX] === '0') {
            if (isCorner(newX, newY) && cells[y][x] !== 'K') {
                break;
            }
            validMoves.push({ x: newX, y: newY });
            newX += dir.x;
            newY += dir.y;
        }
    }
    
    return validMoves;
}


export { HnefataflGame, getValidMoves };


function checkWinCondition(G) {
    // Check if the king is on the board
    const kingExists = G.cells.some(row => row.includes('K'));
    
    // Count the number of attacker pieces
    const attackerCount = G.cells.flat().filter(cell => cell === 'A').length;

    // 1. Attackers win if the king is captured
    if (!kingExists) {
        return 'Attackers';
    }

    // 2. Defenders win conditions
    if (isKingInCorner(G.cells) || (attackerCount === 0)) {
        return 'Defenders';
    }

    return null;  // No winner yet
}

function isKingInCorner(cells) {
    // Check the four corners for the king
    return (cells[0][0] === 'K') || (cells[0][6] === 'K') || (cells[6][0] === 'K') || (cells[6][6] === 'K');
}

function calculateScore(cells) {
    const attackerCount = cells.flat().filter(cell => cell === 'A').length;
    const defenderCount = cells.flat().filter(cell => cell === 'D').length;

    return attackerCount - (defenderCount * 2);
}
