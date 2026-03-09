import styles from './JobCard.module.css';

export default function JobCard({ job }) {
  if (!job) return null;
  const { job_id, status, progress = 0, message, results, output_location } = job;

  const fillMod = status === 'succeeded' ? styles.fillOk
                : status === 'failed'    ? styles.fillErr
                : '';

  return (
    <div className={styles.card}>
      <div className={styles.top}>
        <div className={styles.jobId}>JOB {job_id?.slice(0, 14)}…</div>
        <div className={`${styles.badge} ${styles[status]}`}>
          {status === 'running' && <span className={styles.spinner} />}
          {status.toUpperCase()}
        </div>
      </div>

      <div className={styles.track}>
        <div className={`${styles.fill} ${fillMod}`} style={{ width: `${progress}%` }} />
      </div>

      <div className={styles.msg}>{message || '—'}</div>

      {results && (
        <div className={styles.resultsGrid}>
          {results.balanced_accuracy != null && (
            <div className={styles.resItem}>
              <span className={styles.resVal}>{(results.balanced_accuracy * 100).toFixed(1)}%</span>
              <span className={styles.resLabel}>Bal. Accuracy</span>
            </div>
          )}
          {results.macro_f1 != null && (
            <div className={styles.resItem}>
              <span className={styles.resVal}>{(results.macro_f1 * 100).toFixed(1)}%</span>
              <span className={styles.resLabel}>Macro F1</span>
            </div>
          )}
          {results.total_epochs != null && (
            <div className={styles.resItem}>
              <span className={styles.resVal}>{Number(results.total_epochs).toLocaleString()}</span>
              <span className={styles.resLabel}>Epochs</span>
            </div>
          )}
        </div>
      )}

      {output_location && status === 'succeeded' && (
        <div className={styles.outPath}>→ {output_location}</div>
      )}
    </div>
  );
}
