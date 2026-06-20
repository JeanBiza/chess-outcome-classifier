import { useState, useEffect } from 'react';
import { Chess } from 'chess.js';
import { Chessboard } from 'react-chessboard';

const API = 'http://localhost:8000';

function App() {
  const [game, setGame] = useState(new Chess());
  const [prediction, setPrediction] = useState({ white: 50, black: 50, draw: 0 });
  const [bestMove, setBestMove] = useState(null);
  const [loadingMove, setLoadingMove] = useState(false);

  const fetchPrediction = async (fen) => {
    try {
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen }),
      });
      const data = await res.json();
      setPrediction({
        white: data.white_win_probability,
        black: data.black_win_probability,
        draw: data.draw_probability,
      });
    } catch (err) {
      console.error('predict error:', err);
    }
  };

  const fetchBestMove = async (fen) => {
    setLoadingMove(true);
    setBestMove(null);
    try {
      const res = await fetch(`${API}/best_move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen, depth: 2 }),
      });
      const data = await res.json();
      setBestMove(data.best_move ?? null);
    } catch (err) {
      console.error('best_move error:', err);
    } finally {
      setLoadingMove(false);
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
    setBestMove(null);
    fetchPrediction(gameCopy.fen());
    return true;
  }

  const reset = () => {
    const newGame = new Chess();
    setGame(newGame);
    setBestMove(null);
    fetchPrediction(newGame.fen());
  };

  const turn = game.turn() === 'w' ? 'White' : 'Black';

  return (
    <div style={styles.root}>
      <h2 style={styles.title}>AI Chess Evaluator</h2>

      <div style={styles.layout}>
        <div style={styles.boardWrap}>
          <Chessboard options={{
            position: game.fen(),
            onPieceDrop: onDrop,
            allowDragging: true,
            dragActivationDistance: 0,
            canDragPiece: () => true,
          }} />
        </div>

        <div style={styles.sidebar}>
          <div style={styles.card}>
            <p style={styles.cardTitle}>Neural Network Evaluation</p>
            <EvalRow label="White wins" value={prediction.white} color="#f0f0f0" />
            <EvalRow label="Black wins" value={prediction.black} color="#888" />
            <EvalRow label="Draw"       value={prediction.draw}  color="#aaa" />
          </div>

          <div style={styles.card}>
            <p style={styles.cardTitle}>Best Move <span style={styles.dim}>({turn} to play)</span></p>
            <div style={styles.moveBox}>
              {loadingMove
                ? <span style={styles.dim}>Calculating...</span>
                : bestMove
                  ? <span style={styles.moveText}>{bestMove}</span>
                  : <span style={styles.dim}>—</span>
              }
            </div>
            <button
              style={styles.btn}
              onClick={() => fetchBestMove(game.fen())}
              disabled={loadingMove || game.isGameOver()}
            >
              {loadingMove ? 'Thinking...' : 'Find Best Move'}
            </button>
          </div>

          <button style={styles.resetBtn} onClick={reset}>
            Reset Board
          </button>
        </div>
      </div>
    </div>
  );
}

function EvalRow({ label, value, color }) {
  return (
    <div style={styles.evalRow}>
      <span style={{ color }}>{label}</span>
      <div style={styles.barWrap}>
        <div style={{ ...styles.bar, width: `${value}%`, backgroundColor: color }} />
      </div>
      <span style={{ color, minWidth: 48, textAlign: 'right' }}>{value.toFixed(1)}%</span>
    </div>
  );
}

const styles = {
  root: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '40px 20px',
    fontFamily: "'Segoe UI', sans-serif",
    background: '#1a1a1a',
    minHeight: '100vh',
    color: '#e0e0e0',
  },
  title: {
    fontSize: 22,
    fontWeight: 600,
    letterSpacing: '0.05em',
    marginBottom: 24,
    color: '#fff',
  },
  layout: {
    display: 'flex',
    gap: 24,
    alignItems: 'flex-start',
    flexWrap: 'wrap',
    justifyContent: 'center',
  },
  boardWrap: {
    width: 480,
  },
  sidebar: {
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
    width: 220,
  },
  card: {
    background: '#242424',
    border: '1px solid #333',
    borderRadius: 8,
    padding: '16px 18px',
  },
  cardTitle: {
    fontSize: 11,
    fontWeight: 600,
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    color: '#888',
    margin: '0 0 12px',
  },
  dim: {
    color: '#555',
    fontSize: 11,
    fontWeight: 400,
    textTransform: 'none',
    letterSpacing: 0,
  },
  evalRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    fontSize: 13,
    marginBottom: 8,
  },
  barWrap: {
    flex: 1,
    height: 4,
    background: '#333',
    borderRadius: 2,
    overflow: 'hidden',
  },
  bar: {
    height: '100%',
    borderRadius: 2,
    transition: 'width 0.4s ease',
  },
  moveBox: {
    height: 40,
    display: 'flex',
    alignItems: 'center',
    marginBottom: 12,
  },
  moveText: {
    fontSize: 26,
    fontWeight: 700,
    fontFamily: 'monospace',
    color: '#ff8c00',
    letterSpacing: '0.05em',
  },
  btn: {
    width: '100%',
    padding: '8px 0',
    background: '#ff8c00',
    color: '#000',
    border: 'none',
    borderRadius: 6,
    fontWeight: 700,
    fontSize: 13,
    cursor: 'pointer',
    opacity: 1,
    transition: 'opacity 0.2s',
  },
  resetBtn: {
    width: '100%',
    padding: '8px 0',
    background: 'transparent',
    color: '#555',
    border: '1px solid #333',
    borderRadius: 6,
    fontSize: 13,
    cursor: 'pointer',
  },
};

export default App;