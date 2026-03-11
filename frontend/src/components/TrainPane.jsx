import { useState, useRef, useEffect } from 'react';
import { API } from '../constants';
import JobCard from './JobCard';
import s from './shared.module.css';
import styles from './FormPane.module.css';

const DEFAULT = { npz_dir: '', model_out: '', fs: 100, n_splits: 2, model_type: 'lda' };

export default function TrainPane({ onTrained, npzDir }) {
  const [cfg,     setCfg]     = useState(DEFAULT);
  const [job,     setJob]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);
  const poll = useRef(null);

  const set = (k, v) => setCfg(p => ({ ...p, [k]: v }));
  const stopPoll = () => { clearInterval(poll.current); poll.current = null; };

  async function pollJob(id) {
    try {
      const r = await fetch(`${API}/v1/train/${id}`);
      const d = await r.json();
      setJob(d);
      if (d.status === 'succeeded' || d.status === 'failed') {
        stopPoll();
        if (d.status === 'succeeded') onTrained?.();
      }
    } catch {
      // ignore poll errors
    }
  }

  async function submit() {
    setLoading(true); setError(null); stopPoll();
    const body = {
      ...(npzDir
        ? { npz_dir: npzDir }
        : { npz_dir: cfg.npz_dir }),
      model_out: cfg.model_out || null,
      fs:        parseFloat(cfg.fs),
      n_splits:  parseInt(cfg.n_splits),
      model_type: cfg.model_type,
    };
    try {
      const r = await fetch(`${API}/v1/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || 'Train request failed');
      setJob({ job_id: d.job_id, status: 'queued', progress: 0, message: 'Training queued' });
      poll.current = setInterval(() => pollJob(d.job_id), 2000);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => () => stopPoll(), []);

  return (
    <div className={styles.pane}>
      <div className={s.card}>
        <div className={s.cardTitle}>Train Classifier</div>

        {npzDir && (
          <div className={s.sessionBanner}>
            Using preprocessed output — NPZ directory resolved automatically.
          </div>
        )}

        <div className={s.grid2}>
          <div className={`${s.field} ${s.full}`}>
            <label className={s.label}>Model Architecture</label>
            <select 
              value={cfg.model_type} 
              onChange={e => set('model_type', e.target.value)}
              style={{ width: '100%', padding: '0.5rem', background: 'var(--bg-el)', color: 'var(--fg)', border: '1px solid var(--border)', borderRadius: '4px' }}
            >
              <option value="lda">Linear Discriminant Analysis (LDA)</option>
              <option value="random_forest">Random Forest (RF)</option>
            </select>
          </div>
          {!npzDir && (
            <div className={`${s.field} ${s.full}`}>
              <label className={s.label}>NPZ Directory</label>
              <input type="text" value={cfg.npz_dir}
                onChange={e => set('npz_dir', e.target.value)}
                placeholder="/tmp/preprocessed" />
            </div>
          )}
          <div className={`${s.field} ${s.full}`}>
            <label className={s.label}>
              Model Output Path <span className={s.labelNote}>optional — blank = auto</span>
            </label>
            <input type="text" value={cfg.model_out}
              onChange={e => set('model_out', e.target.value)}
              placeholder="/models/lda_baseline.joblib" />
          </div>
          <div className={s.field}>
            <label className={s.label}>Sampling Rate (Hz)</label>
            <input type="number" value={cfg.fs} min={1} step={1}
              onChange={e => set('fs', e.target.value)} />
          </div>
          <div className={s.field}>
            <label className={s.label}>CV Folds (n_splits)</label>
            <input type="number" value={cfg.n_splits} min={2}
              onChange={e => set('n_splits', e.target.value)} />
          </div>
        </div>

        {error && <div className={s.errStrip}>{error}</div>}

        <button
          className={`${s.btn} ${s.btnPrimary}`}
          onClick={submit}
          disabled={(!npzDir && !cfg.npz_dir) || loading}
        >
          {loading && <span className={s.spinner} />}
          {loading ? 'Submitting…' : 'Start Training'}
        </button>
      </div>

      <JobCard job={job} />
    </div>
  );
}