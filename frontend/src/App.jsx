import { useState, useEffect } from 'react';
import { Chess } from 'chess.js';
import { Chessboard } from 'react-chessboard';

function App() {
  const [game, setGame] = useState(new Chess());
  const [prediction, setPrediction] = useState({ white: 50, black: 50, draw: 0 });

  const fetchPrediction = async (fen) => {
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen }),
      });
      const data = await response.json();
      setPrediction({
        white: data.white_win_probability,
        black: data.black_win_probability,
        draw: data.draw_probability,
      });
    } catch (error) {
      console.error("API error:", error);
    }
  };

  useEffect(() => {
    fetchPrediction(game.fen());
  }, []);

  function onDrop({ sourceSquare, targetSquare }) {
    const gameCopy = new Chess();
    gameCopy.load(game.fen());

    const move = gameCopy.move({ from: sourceSquare, to: targetSquare, promotion: 'q' });
    if (move === null) return false;

    setGame(gameCopy);
    fetchPrediction(gameCopy.fen());
    return true;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: '50px', fontFamily: 'sans-serif' }}>
      <h2>AI Chess Endgame Evaluator</h2>

      <div style={{ width: '500px', marginBottom: '20px' }}>
        <Chessboard options={{
          position: game.fen(),
          onPieceDrop: onDrop,
          allowDragging: true,
          dragActivationDistance: 0,
          canDragPiece: () => true,
        }} />
      </div>

      <div style={{ padding: '20px', border: '1px solid #ccc', borderRadius: '8px', width: '460px', textAlign: 'center' }}>
        <h3>Neural Network Evaluation</h3>
        <p><strong>White wins:</strong> {prediction.white.toFixed(2)}%</p>
        <p><strong>Black wins:</strong> {prediction.black.toFixed(2)}%</p>
        <p><strong>Draw:</strong> {prediction.draw.toFixed(2)}%</p>
      </div>

      <div style={{ marginTop: '20px' }}>
        <button
          onClick={() => {
            const newGame = new Chess();
            setGame(newGame);
            fetchPrediction(newGame.fen());
          }}
          style={{ padding: '10px 20px', cursor: 'pointer', borderRadius: '5px', border: '1px solid #ccc' }}
        >
          Reset Board
        </button>
      </div>
    </div>
  );
}

export default App;