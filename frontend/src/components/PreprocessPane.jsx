import { useState, useRef, useEffect } from 'react';
import { API } from '../constants';
import JobCard from './JobCard';
import s from './shared.module.css';
import styles from './FormPane.module.css';

const DEFAULT = {
  raw_dir: '', out_dir: '',
  channel: 'EEG Fpz-Cz',
  epochs: 30,
  bp_lo: 0.3, bp_hi: 30.0,
  notch: '',
  combine: true,
  dry_run: false,
};

export default function PreprocessPane({ sessionId, onDone }) {
  const [cfg,     setCfg]     = useState(DEFAULT);
  const [job,     setJob]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);
  const poll = useRef(null);

  const set = (k, v) => setCfg(p => ({ ...p, [k]: v }));
  const stopPoll = () => { clearInterval(poll.current); poll.current = null; };

  async function pollJob(id) {
    try {
      const r = await fetch(`${API}/v1/preprocess/${id}`);
      const d = await r.json();
      setJob(d);
      if (d.status === 'succeeded' || d.status === 'failed') {
        stopPoll();
        if (d.status === 'succeeded') onDone?.(d.output_location);
      }
    } catch {
      // ignore poll errors
    }
  }

  async function submit() {
    setLoading(true); setError(null); stopPoll();
    const body = {
      ...(sessionId
        ? { session_id: sessionId }
        : { dataset: { type: 'local_edf', raw_dir: cfg.raw_dir } }),
      output:  { out_dir: cfg.out_dir || null, combine: cfg.combine },
      preprocessing_config: {
        channel:  cfg.channel,
        epochs:   parseInt(cfg.epochs),
        bandpass: [parseFloat(cfg.bp_lo), parseFloat(cfg.bp_hi)],
        notch:    cfg.notch ? parseFloat(cfg.notch) : null,
      },
      dry_run: cfg.dry_run,
    };
    try {
      const r = await fetch(`${API}/v1/preprocess`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || 'Request failed');

      if (cfg.dry_run) {
        setJob({ job_id: 'dry-run', status: 'succeeded', progress: 100, message: d.message });
      } else {
        setJob({ job_id: d.job_id, status: 'queued', progress: 0, message: 'Job submitted' });
        poll.current = setInterval(() => pollJob(d.job_id), 1600);
      }
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
        <div className={s.cardTitle}>Preprocess EDF Data</div>

        {sessionId && (
          <div className={s.sessionBanner}>
            Using uploaded session — raw directory resolved automatically.
          </div>
        )}

        <div className={s.grid2}>
          {!sessionId && (
            <div className={`${s.field} ${s.full}`}>
              <label className={s.label}>Raw EDF Directory</label>
              <input type="text" value={cfg.raw_dir}
                onChange={e => set('raw_dir', e.target.value)}
                placeholder="/data/raw/sleep-edf" />
            </div>
          )}
          <div className={`${s.field} ${s.full}`}>
            <label className={s.label}>
              Output Directory <span className={s.labelNote}>(blank = auto-generated)</span>
            </label>
            <input type="text" value={cfg.out_dir}
              onChange={e => set('out_dir', e.target.value)}
              placeholder="Leave blank for temp dir" />
          </div>
          <div className={s.field}>
            <label className={s.label}>EEG Channel</label>
            <input type="text" value={cfg.channel}
              onChange={e => set('channel', e.target.value)} />
          </div>
          <div className={s.field}>
            <label className={s.label}>Epoch Length (s)</label>
            <input type="number" value={cfg.epochs} min={1}
              onChange={e => set('epochs', e.target.value)} />
          </div>
          <div className={s.field}>
            <label className={s.label}>Bandpass Low (Hz)</label>
            <input type="number" value={cfg.bp_lo} step={0.1}
              onChange={e => set('bp_lo', e.target.value)} />
          </div>
          <div className={s.field}>
            <label className={s.label}>Bandpass High (Hz)</label>
            <input type="number" value={cfg.bp_hi} step={0.1}
              onChange={e => set('bp_hi', e.target.value)} />
          </div>
          <div className={s.field}>
            <label className={s.label}>
              Notch Filter (Hz) <span className={s.labelNote}>optional</span>
            </label>
            <input type="number" value={cfg.notch} placeholder="50 or 60"
              onChange={e => set('notch', e.target.value)} />
          </div>
        </div>

        <div className={styles.toggleRow}>
          <label className={s.toggle}>
            <input className={s.toggleInput} type="checkbox" checked={cfg.combine}
              onChange={e => set('combine', e.target.checked)} />
            <div className={s.toggleTrack}><div className={s.toggleThumb} /></div>
            <span className={s.toggleLabel}>Combine nights</span>
          </label>
          <label className={s.toggle}>
            <input className={s.toggleInput} type="checkbox" checked={cfg.dry_run}
              onChange={e => set('dry_run', e.target.checked)} />
            <div className={s.toggleTrack}><div className={s.toggleThumb} /></div>
            <span className={s.toggleLabel}>Dry run (validate only)</span>
          </label>
        </div>

        {error && <div className={s.errStrip}>{error}</div>}

        <button
          className={`${s.btn} ${s.btnPrimary}`}
          onClick={submit}
          disabled={(!sessionId && !cfg.raw_dir) || loading}
        >
          {loading && <span className={s.spinner} />}
          {loading ? 'Submitting…' : cfg.dry_run ? 'Validate Only' : 'Run Preprocessing'}
        </button>
      </div>

      <JobCard job={job} />
    </div>
  );
}
