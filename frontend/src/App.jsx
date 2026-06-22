import { useState, useEffect, useRef } from 'react';
import { Chess } from 'chess.js';
import { Chessboard } from 'react-chessboard';

const API = 'http://localhost:8000';

function App() {
  const [game, setGame]             = useState(new Chess());
  const [prediction, setPrediction] = useState({ white: 50, black: 50, draw: 0 });
  const [bestMove, setBestMove]     = useState(null);
  const [loadingMove, setLoadingMove] = useState(false);
  const [arrow, setArrow]           = useState([]);
  const [orientation, setOrientation] = useState('white');
  const [history, setHistory]       = useState([]);
  const [gameOver, setGameOver]     = useState(null);
  const [copyFeedback, setCopyFeedback] = useState(null);
  const historyRef = useRef(null);

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
    setArrow([]);
    try {
      const res = await fetch(`${API}/best_move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen, depth: 2 }),
      });
      const data = await res.json();
      if (data.best_move) {
        setBestMove(data.best_move);
        setArrow([{
          startSquare: data.best_move.slice(0, 2),
          endSquare:   data.best_move.slice(2, 4),
          color: 'orange',
        }]);
      }
    } catch (err) {
      console.error('best_move error:', err);
    } finally {
      setLoadingMove(false);
    }
  };

  useEffect(() => {
    fetchPrediction(game.fen());
  }, []);

  useEffect(() => {
    if (historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight;
    }
  }, [history]);

  function checkGameOver(g) {
    if (!g.isGameOver()) return null;
    if (g.isCheckmate()) return `${g.turn() === 'w' ? 'Black' : 'White'} wins by checkmate`;
    if (g.isStalemate()) return 'Draw by stalemate';
    if (g.isThreefoldRepetition()) return 'Draw by repetition';
    if (g.isInsufficientMaterial()) return 'Draw by insufficient material';
    if (g.isDraw()) return 'Draw by 50-move rule';
    return 'Game over';
  }

  function onDrop({ sourceSquare, targetSquare }) {
    if (gameOver) return false;
    const gameCopy = new Chess();
    gameCopy.load(game.fen());
    const move = gameCopy.move({ from: sourceSquare, to: targetSquare, promotion: 'q' });
    if (move === null) return false;

    setGame(gameCopy);
    setHistory(h => [...h, move.san]);
    setArrow([]);
    setBestMove(null);

    const over = checkGameOver(gameCopy);
    if (over) setGameOver(over);
    else fetchPrediction(gameCopy.fen());
    return true;
  }

  const undoMove = () => {
    if (history.length === 0 || gameOver) return;
    
    const gameCopy = new Chess();
    const movesToReplay = history.slice(0, -1);
    movesToReplay.forEach(san => gameCopy.move(san));
    
    setGame(gameCopy);
    setHistory(h => h.slice(0, -1));
    setArrow([]);
    setBestMove(null);
    fetchPrediction(gameCopy.fen());
  };

  const reset = () => {
    const newGame = new Chess();
    setGame(newGame);
    setHistory([]);
    setArrow([]);
    setBestMove(null);
    setGameOver(null);
    setCopyFeedback(null);
    fetchPrediction(newGame.fen());
  };

  const flipBoard = () => setOrientation(o => o === 'white' ? 'black' : 'white');

  const copyToClipboard = async (text, key) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopyFeedback(key);
      setTimeout(() => setCopyFeedback(null), 2000);
    } catch (err) {
      console.error('Copy failed:', err);
    }
  };

  const turn = game.turn() === 'w' ? 'White' : 'Black';

  const movePairs = [];
  for (let i = 0; i < history.length; i += 2) {
    movePairs.push([history[i], history[i + 1]]);
  }

  return (
    <div style={styles.root}>
      <h2 style={styles.title}>AI Chess Evaluator</h2>

      <div style={styles.layout}>
        <div style={styles.boardWrap}>
          <Chessboard options={{
            position: game.fen(),
            onPieceDrop: onDrop,
            allowDragging: !gameOver,
            dragActivationDistance: 0,
            canDragPiece: () => !gameOver,
            arrows: arrow,
            boardOrientation: orientation,
          }} />
        </div>

        <div style={styles.sidebar}>

          {gameOver && (
            <div style={styles.gameOverCard}>
              <div style={styles.gameOverHeader}>
                <span style={styles.gameOverIcon}>♟</span>
                <div style={{ display: 'flex', gap: 6 }}>
                  <button
                    style={styles.copyBtn}
                    onClick={() => copyToClipboard(game.pgn(), 'pgn')}
                    title="Copy PGN"
                  >
                    {copyFeedback === 'pgn' ? '✓' : 'PGN'}
                  </button>
                  <button
                    style={styles.copyBtn}
                    onClick={() => copyToClipboard(game.fen(), 'fen')}
                    title="Copy FEN"
                  >
                    {copyFeedback === 'fen' ? '✓' : 'FEN'}
                  </button>
                </div>
              </div>
              <p style={styles.gameOverText}>{gameOver}</p>
            </div>
          )}

          <div style={styles.card}>
            <p style={styles.cardTitle}>Neural Network Evaluation</p>
            <EvalRow label="White wins" value={prediction.white} color="#f0f0f0" />
            <EvalRow label="Black wins" value={prediction.black} color="#888" />
            <EvalRow label="Draw"       value={prediction.draw}  color="#555" />
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
              disabled={loadingMove || !!gameOver}
            >
              {loadingMove ? 'Thinking...' : 'Find Best Move'}
            </button>
          </div>

          <div style={styles.card}>
            <p style={styles.cardTitle}>Move History</p>
            <div style={styles.historyBox} ref={historyRef}>
              {movePairs.length === 0
                ? <span style={styles.dim}>No moves yet</span>
                : movePairs.map(([w, b], i) => (
                  <div key={i} style={styles.moveRow}>
                    <span style={styles.moveNum}>{i + 1}.</span>
                    <span style={styles.moveSan}>{w}</span>
                    <span style={{ ...styles.moveSan, color: b ? '#aaa' : 'transparent' }}>{b ?? '...'}</span>
                  </div>
                ))
              }
            </div>
          </div>

          <div style={styles.buttonRow}>
            <button style={styles.resetBtn} onClick={reset}>Reset</button>
            <button style={styles.resetBtn} onClick={flipBoard}>Flip</button>
            <button
              style={{ ...styles.resetBtn, opacity: history.length === 0 || !!gameOver ? 0.3 : 1 }}
              onClick={undoMove}
              disabled={history.length === 0 || !!gameOver}
            >
              Undo
            </button>
          </div>

        </div>
      </div>
    </div>
  );
}

function EvalRow({ label, value, color }) {
  return (
    <div style={styles.evalRow}>
      <span style={{ color, minWidth: 72, fontSize: 12 }}>{label}</span>
      <div style={styles.barWrap}>
        <div style={{ ...styles.bar, width: `${value}%`, backgroundColor: color }} />
      </div>
      <span style={{ color, minWidth: 44, textAlign: 'right', fontSize: 12 }}>{value.toFixed(1)}%</span>
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
  boardWrap: { width: 480 },
  sidebar: {
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
    width: 220,
  },
  gameOverCard: {
    background: '#2a1f0e',
    border: '1px solid #ff8c00',
    borderRadius: 8,
    padding: '12px 14px',
  },
  gameOverHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  gameOverIcon: { fontSize: 24 },
  copyBtn: {
    background: '#3a2810',
    border: '1px solid #ff8c00',
    color: '#ff8c00',
    fontSize: 11,
    fontWeight: 700,
    cursor: 'pointer',
    padding: '2px 7px',
    borderRadius: 4,
    letterSpacing: '0.05em',
  },
  gameOverText: {
    margin: 0,
    fontWeight: 700,
    color: '#ff8c00',
    fontSize: 13,
  },
  card: {
    background: '#242424',
    border: '1px solid #333',
    borderRadius: 8,
    padding: '14px 16px',
  },
  cardTitle: {
    fontSize: 11,
    fontWeight: 600,
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    color: '#888',
    margin: '0 0 10px',
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
    marginBottom: 7,
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
    height: 38,
    display: 'flex',
    alignItems: 'center',
    marginBottom: 10,
  },
  moveText: {
    fontSize: 24,
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
  },
  historyBox: {
    maxHeight: 160,
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
  },
  moveRow: {
    display: 'flex',
    gap: 8,
    fontSize: 13,
    fontFamily: 'monospace',
  },
  moveNum: { color: '#555', minWidth: 22 },
  moveSan: { color: '#e0e0e0', minWidth: 48 },
  buttonRow: {
    display: 'flex',
    gap: 8,
  },
  resetBtn: {
    flex: 1,
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