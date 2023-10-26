const HnefataflBoard = ({ ctx, G, moves }) => {

    let gameover = ctx.gameover;
    let displayMessage;
    if (gameover) {
        displayMessage = `Winner: ${gameover.winner}`;
    } else {
        displayMessage = `Score: ${G.score}`;
    }
    
    return (
        <div>
            <div className="score-display">{displayMessage}</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 40px)' }}>
                {G.cells.flat().map((cell, idx) => {
                    const x = idx % 7;
                    const y = Math.floor(idx / 7);
                    const isSelected = G.selected && G.selected.x === x && G.selected.y === y;
                    const isValidMove = G.validMoves.some(move => move.x === x && move.y === y);
        
                    return (
                        <div
                            key={idx}
                            style={{
                                width: '40px',
                                height: '40px',
                                border: '1px solid black',
                                textAlign: 'center',
                                lineHeight: '40px',
                                backgroundColor: isSelected ? 'yellow' : isValidMove ? 'lightgreen' : 'white'
                            }}
                            onClick={() => {                   
                                console.log("From Board:");
                                console.log(G);                
                                console.log(ctx);                
                                if (cell !== '0' && !isSelected) {
                                    moves.selectPiece(x, y);
                                } else if (isValidMove) {
                                    moves.movePiece(x, y);
                                }
                            }}
                        >
                            {cell}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export { HnefataflBoard };
