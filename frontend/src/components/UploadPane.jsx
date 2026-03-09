import { useState, useRef } from 'react';
import { API } from '../constants';
import s from './shared.module.css';
import styles from './UploadPane.module.css';

export default function UploadPane({ onDone }) {
  const [files,   setFiles]   = useState([]);
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);
  const [drag,    setDrag]    = useState(false);
  const inp = useRef(null);

  const pick = (fs) => { setFiles(Array.from(fs)); setResult(null); setError(null); };

  async function submit() {
    setLoading(true); setError(null);
    const form = new FormData();
    files.forEach(f => form.append('files', f));
    try {
      const r = await fetch(`${API}/v1/upload`, { method: 'POST', body: form });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || 'Upload failed');
      setResult(d);
      onDone?.(d);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.pane}>
      <div className={s.card}>
        <div className={s.cardTitle}>Upload EDF Files</div>

        <div
          className={`${styles.dropzone} ${drag ? styles.dragOver : ''}`}
          onClick={() => inp.current?.click()}
          onDragOver={e => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={e => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files); }}
        >
          <input
            ref={inp}
            type="file"
            multiple
            accept=".edf"
            style={{ display: 'none' }}
            onChange={e => pick(e.target.files)}
          />
          <svg className={styles.icon} viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="8" y="14" width="32" height="26" rx="2" />
            <path d="M24 8v18M17 15l7-7 7 7" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <div className={styles.dropText}>Drop EDF files here or click to browse</div>
          <div className={styles.dropSub}>Requires *PSG.edf + *Hypnogram.edf pairs</div>
        </div>

        {files.length > 0 && (
          <div className={styles.fileList}>
            {files.map((f, i) => (
              <div key={i} className={styles.fileItem}>
                <span className={`${styles.ftag} ${f.name.includes('Hypnogram') ? styles.hyp : styles.psg}`}>
                  {f.name.includes('Hypnogram') ? 'HYP' : 'PSG'}
                </span>
                <span className={styles.fileName}>{f.name}</span>
                <span className={styles.fileSize}>{(f.size / 1024 / 1024).toFixed(1)} MB</span>
              </div>
            ))}
          </div>
        )}

        {error && <div className={`${s.errStrip} ${styles.errMargin}`}>{error}</div>}

        <div className={s.btnRow}>
          <button className={`${s.btn} ${s.btnPrimary}`} onClick={submit} disabled={!files.length || loading}>
            {loading && <span className={s.spinner} />}
            {loading ? 'Uploading…' : 'Upload Files'}
          </button>
          {files.length > 0 && (
            <button className={`${s.btn} ${s.btnGhost}`} onClick={() => { setFiles([]); setResult(null); }}>
              Clear
            </button>
          )}
        </div>
      </div>

      {result && (
        <div className={`${s.card} ${s.cardSuccess}`}>
          <div className={`${s.cardTitle} ${s.cardTitleSuccess}`}>Upload Successful</div>
          <div className={styles.sessionId}>Session: {result.session_id}</div>
          <div className={styles.resultMsg}>{result.message}</div>
          <div className={styles.fileList}>
            {result.files?.map((f, i) => (
              <div key={i} className={styles.fileItem}>
                <span className={`${styles.ftag} ${styles.psg}`}>PSG</span>
                <span className={styles.fileName}>{f.psg_file}</span>
                <span className={styles.plus}>+</span>
                <span className={`${styles.ftag} ${styles.hyp}`}>HYP</span>
                <span className={styles.fileName}>{f.hypnogram_file}</span>
              </div>
            ))}
          </div>
          <div className={styles.rawDir}>RAW DIR: {result.raw_dir}</div>
        </div>
      )}
    </div>
  );
}
