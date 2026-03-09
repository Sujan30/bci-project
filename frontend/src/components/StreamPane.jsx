import { useState, useRef, useEffect } from 'react';
import { WS, STAGES, STAGE_ORDER, DEMO_SEQ } from '../constants';
import s from './shared.module.css';
import styles from './StreamPane.module.css';

function mkDemoFrame(seqRef) {
  const key = DEMO_SEQ[seqRef.current++ % DEMO_SEQ.length];
  const idx = STAGE_ORDER.indexOf(key);
  const raw = STAGE_ORDER.map((_, i) =>
    i === idx ? 0.55 + Math.random() * 0.35 : Math.random() * 0.12
  );
  const sum  = raw.reduce((a, b) => a + b, 0);
  const prob = Object.fromEntries(STAGE_ORDER.map((k, i) => [k, raw[i] / sum]));
  return {
    stage:      key,
    probas:     prob,
    latency_ms: parseFloat((Math.random() * 3 + 0.5).toFixed(2)),
  };
}

export default function StreamPane({ models }) {
  const [modelId,    setModelId]    = useState('');
  const [demo,       setDemo]       = useState(true);
  const [connected,  setConnected]  = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [wsErr,      setWsErr]      = useState(null);
  const [stage,      setStage]      = useState(null);
  const [probas,     setProbas]     = useState(null);
  const [latency,    setLatency]    = useState(null);
  const [history,    setHistory]    = useState([]);

  const wsRef   = useRef(null);
  const demoRef = useRef(null);
  const seqRef  = useRef(0);

  function applyFrame(f) {
    setStage(f.stage);
    setProbas(f.probas);
    setLatency(f.latency_ms);
    setHistory(h => [...h.slice(-79), f.stage]);
  }

  /* Demo mode */
  function startDemo() {
    seqRef.current = 0;
    demoRef.current = setInterval(() => applyFrame(mkDemoFrame(seqRef)), 1100);
    setConnected(true); setWsErr(null);
  }
  function stopDemo() {
    clearInterval(demoRef.current);
    setConnected(false); setStage(null); setProbas(null); setHistory([]);
  }

  /* Real WS */
  function connectWS() {
    if (!modelId) return;
    setConnecting(true); setWsErr(null);
    const ws = new WebSocket(`${WS}/v1/stream?model_id=${encodeURIComponent(modelId)}`);
    wsRef.current = ws;
    ws.onopen    = () => { setConnected(true); setConnecting(false); sendEpoch(ws); };
    ws.onmessage = e => {
      const d = JSON.parse(e.data);
      if (d.error) { setWsErr(d.message || d.error); return; }
      applyFrame({
        stage:      d.stage,
        probas:     buildFakeProbas(d.stage, d.confidence),
        latency_ms: d.latency_ms,
      });
      sendEpoch(ws);
    };
    ws.onerror = () => { setWsErr('WebSocket error'); setConnecting(false); setConnected(false); };
    ws.onclose = () => { setConnected(false); setConnecting(false); };
  }
  function disconnectWS() {
    wsRef.current?.close();
    setConnected(false); setStage(null); setProbas(null); setHistory([]);
  }
  function sendEpoch(ws) {
    const target = ws || wsRef.current;
    if (target?.readyState === 1) {
      const epoch = Array.from({ length: 3000 }, () => (Math.random() - 0.5) * 200);
      target.send(JSON.stringify({ epoch, timestamp: Date.now() }));
    }
  }
  function buildFakeProbas(stg, conf) {
    const map = Object.fromEntries(STAGE_ORDER.map(k => [k, 0]));
    map[stg] = conf || 1;
    return map;
  }

  useEffect(() => () => { clearInterval(demoRef.current); wsRef.current?.close(); }, []);

  const toggleDemo = val => {
    setDemo(val);
    if (connected) {
      if (val) { disconnectWS(); startDemo(); }
      else     { stopDemo(); }
    }
  };

  const stg   = stage ? STAGES[stage] : null;
  const color = stg?.color ?? 'var(--text-dim)';

  return (
    <div className={styles.pane}>
      {/* Controls */}
      <div className={s.card}>
        <div className={s.cardTitle}>Live Inference Stream</div>
        <div className={styles.controls}>
          <label className={s.toggle}>
            <input className={s.toggleInput} type="checkbox" checked={demo}
              onChange={e => toggleDemo(e.target.checked)} />
            <div className={s.toggleTrack}><div className={s.toggleThumb} /></div>
            <span className={s.toggleLabel}>Demo Mode</span>
          </label>

          {!demo && (
            <input
              type="text"
              style={{ flex: 1, maxWidth: 220 }}
              value={modelId}
              placeholder="model_id (e.g. lda_baseline)"
              onChange={e => setModelId(e.target.value)}
            />
          )}

          {connected ? (
            <button className={`${s.btn} ${s.btnDanger}`}
              onClick={demo ? stopDemo : disconnectWS}>
              ■&ensp;Disconnect
            </button>
          ) : (
            <button
              className={`${s.btn} ${s.btnPrimary}`}
              onClick={demo ? startDemo : connectWS}
              disabled={connecting || (!demo && !modelId)}
            >
              {connecting && <span className={s.spinner} />}
              {connecting ? 'Connecting…' : '▶\u2002Connect'}
            </button>
          )}

          <div className={`${styles.wsState} ${connected ? styles.wsOn : wsErr ? styles.wsErr : ''}`}>
            <div className={`${styles.wsDot} ${connected ? styles.wsDotOn : wsErr ? styles.wsDotErr : ''}`} />
            {connected ? 'STREAMING' : wsErr ? 'ERROR' : 'IDLE'}
          </div>
        </div>

        {wsErr && <div className={s.errStrip}>{wsErr}</div>}
      </div>

      {connected && (
        <>
          {/* Stage + probabilities */}
          <div className={styles.streamCols}>
            {/* Big stage display */}
            <div className={styles.stageDisplay}>
              <div className={styles.stageBgGlow} style={{
                boxShadow: stg
                  ? `inset 0 0 90px ${color}12, 0 0 0 1px ${color}30`
                  : 'none',
              }} />
              <div className={styles.stageBig} style={{ color }}>
                {stg?.label ?? '—'}
              </div>
              <div className={styles.stageDescLabel}>{stg?.desc ?? 'Awaiting signal…'}</div>
              <div className={styles.stageConf} style={{ color }}>
                {probas && stage ? `${(probas[stage] * 100).toFixed(1)}%` : '—'}
              </div>
              <div className={styles.stageLatency}>
                {latency ? `${latency} ms` : ''}
              </div>
            </div>

            {/* Probability bars */}
            <div className={styles.probaCard}>
              <div className={s.cardTitle} style={{ marginBottom: 14 }}>Stage Probabilities</div>
              {STAGE_ORDER.map(sk => (
                <div key={sk} className={styles.probaRow}>
                  <div className={styles.probaName} style={{ color: STAGES[sk].color }}>{sk}</div>
                  <div className={styles.probaTrack}>
                    <div
                      className={styles.probaFill}
                      style={{
                        width:     `${((probas?.[sk] ?? 0) * 100).toFixed(1)}%`,
                        background: STAGES[sk].color,
                        boxShadow:  stage === sk ? `0 0 7px ${STAGES[sk].color}` : 'none',
                      }}
                    />
                  </div>
                  <div className={styles.probaPct}>
                    {probas ? `${(probas[sk] * 100).toFixed(1)}%` : '—'}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Hypnogram */}
          <div className={styles.hypnogram}>
            <div className={s.cardTitle} style={{ marginBottom: 14 }}>
              <span>Session Hypnogram</span>
              <span className={styles.epochCount}>{history.length} epochs</span>
            </div>
            <div className={styles.hypnoTrack}>
              {history.length === 0 ? (
                <div className={styles.hypnoEmpty}>Awaiting epochs…</div>
              ) : history.map((hs, i) => (
                <div
                  key={i}
                  className={styles.hypnoBlock}
                  style={{ background: STAGES[hs]?.color ?? '#333' }}
                  title={hs}
                />
              ))}
            </div>
            <div className={styles.hypnoLegend}>
              {STAGE_ORDER.map(sk => (
                <div key={sk} className={styles.legendItem}>
                  <div className={styles.legendSwatch} style={{ background: STAGES[sk].color }} />
                  <span className={styles.legendName}>{sk}</span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}