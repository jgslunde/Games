const HnefataflBoard = ({ ctx, G, events, moves }) => {

    const handleRestart = () => {
        events.endGame();
        // events.restart();
    };
    let gameover = ctx.gameover;
    let displayMessage;
    if (gameover) {
        displayMessage = (
            <div>
                <div>Winner: {gameover.winner}</div>
                <button onClick={handleRestart}>Play Again</button>
            </div>
        );
    } else {
        displayMessage = (
            <div>
                <div>Current score: {G.score}</div>
                <div>To play: {ctx.currentPlayer==='0' ? "Attacker" : "Defender"}</div>
            </div>
        )
    }

    const isPlayablePiece = (piece) => {
        if ((ctx.currentPlayer === '0') && (piece === 'A')) {
            return true;
        } else if ((ctx.currentPlayer === '1') && ((piece === 'D') || (piece === 'K'))) {
            return true;
        }
        return false;
    };
    
    return (
        <div className="container">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 120px)' }}> {/* Adjusted grid column size */}
                {G.cells.flat().map((cell, idx) => {
                    const x = idx % 7;
                    const y = Math.floor(idx / 7);
                    const isEven = (x + y) % 2 === 0;
                    const isCorner = ((x === 0) || (x === 6)) && ((y === 0) || (y === 6));
                    let squareColor = isEven ? '#cecece' : '#8e8e8e';
                    if(isCorner)
                        squareColor = '#353535';
                    const isSelected = G.selected && G.selected.x === x && G.selected.y === y;
                    const isValidMove = G.validMoves.some(move => move.x === x && move.y === y);
                    let displaySymbol = cell;  // Default to the cell value
                    if (cell === 'K') displaySymbol = '🌍';//'⬛'//'♚';
                    if (cell === 'D') displaySymbol = '🌕'; //'⚫';
                    if (cell === 'A') displaySymbol = '🌑';//'⚪';
                    if (cell === '0') displaySymbol = '';

                    return (
                        <div
                            key={idx}
                            style={{
                                width: '120px',  // Adjusted width
                                height: '120px',  // Adjusted height
                                border: '0px solid black',
                                textAlign: 'center',
                                lineHeight: '120px',  // Adjusted line height
                                backgroundColor: isSelected ? "rgba(100, 200, 0, 1.0)" : isValidMove ? "rgba(50, 150, 0, 1.0)" : squareColor,  // Integrated square color
                                fontSize: '60px',
                                cursor: isPlayablePiece(cell) ? 'pointer' : 'default',
                            }}
                            onClick={() => {                   
                                if (cell !== '0' && !isSelected) {
                                    moves.selectPiece(x, y);
                                } else if (isValidMove) {
                                    moves.movePiece(x, y);
                                } else if (isSelected) {
                                    moves.deselectPiece();  // Use the move function for deselecting
                                }
                            }}
                        >
                            {displaySymbol}
                        </div>
                    );
                })}
            </div>
            <div className="score-display" style={{fontSize : "32px"}}> {displayMessage}</div>
        </div>
    );
};

export { HnefataflBoard };
