import styles from './JobCard.module.css';

const STAGE_COLORS = {
  W:   'var(--wake)',
  N1:  'var(--n1)',
  N2:  'var(--n2)',
  N3:  'var(--n3)',
  REM: 'var(--rem)',
};
const STAGE_ORDER = ['W', 'N1', 'N2', 'N3', 'REM'];

function computeAvgF1(folds) {
  const avgF1 = {};
  STAGE_ORDER.forEach(s => {
    const vals = folds.map(f => f.per_class_f1?.[s]).filter(v => v != null);
    avgF1[s] = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  });
  return avgF1;
}

export default function JobCard({ job }) {
  if (!job) return null;
  const { job_id, status, progress = 0, message, results, output_location } = job;

  const fillMod = status === 'succeeded' ? styles.fillOk
                : status === 'failed'    ? styles.fillErr
                : '';

  const overall = results?.overall;
  const balAcc  = overall?.balanced_accuracy_mean;
  const balStd  = overall?.balanced_accuracy_std;
  const macroF1 = overall?.macro_f1_mean;
  const macroStd = overall?.macro_f1_std;

  const avgF1 = results?.folds ? computeAvgF1(results.folds) : null;
  const cm = results?.confusion_matrix;

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

      {results && status === 'succeeded' && (
        <div className={styles.analytics}>

          {/* Part A — Summary stats */}
          <div>
            <div className={styles.sectionLabel}>Summary</div>
            <div className={styles.resultsGrid}>
              {balAcc != null && (
                <div className={styles.resItem}>
                  <span className={styles.resVal}>{(balAcc * 100).toFixed(1)}%</span>
                  {balStd != null && (
                    <span className={styles.resStd}>±{(balStd * 100).toFixed(1)}%</span>
                  )}
                  <span className={styles.resLabel}>Bal. Accuracy</span>
                </div>
              )}
              {macroF1 != null && (
                <div className={styles.resItem}>
                  <span className={styles.resVal}>{(macroF1 * 100).toFixed(1)}%</span>
                  {macroStd != null && (
                    <span className={styles.resStd}>±{(macroStd * 100).toFixed(1)}%</span>
                  )}
                  <span className={styles.resLabel}>Macro F1</span>
                </div>
              )}
              {results.total_epochs != null && (
                <div className={styles.resItem}>
                  <span className={styles.resVal}>{Number(results.total_epochs).toLocaleString()}</span>
                  <span className={styles.resLabel}>Epochs</span>
                </div>
              )}
              {results.total_nights != null && (
                <div className={styles.resItem}>
                  <span className={styles.resVal}>{results.total_nights}</span>
                  <span className={styles.resLabel}>Nights</span>
                </div>
              )}
            </div>
          </div>

          {/* Part B — Per-class F1 bars */}
          {avgF1 && (
            <div>
              <div className={styles.sectionLabel}>Per-class F1</div>
              {STAGE_ORDER.map(s => {
                const val = avgF1[s];
                if (val == null) return null;
                return (
                  <div key={s} className={styles.f1Row}>
                    <span className={styles.stageChip} style={{ color: STAGE_COLORS[s] }}>{s}</span>
                    <div className={styles.f1Track}>
                      <div
                        className={styles.f1Bar}
                        style={{ width: `${(val * 100).toFixed(1)}%`, background: STAGE_COLORS[s] }}
                      />
                    </div>
                    <span className={styles.f1Pct}>{(val * 100).toFixed(1)}%</span>
                  </div>
                );
              })}
            </div>
          )}

          {/* Part C — Confusion matrix heatmap */}
          {cm && cm.labels && cm.matrix && cm.normalized && (
            <div>
              <div className={styles.sectionLabel}>Confusion Matrix</div>
              <div className={styles.cmWrapper}>
                <div className={styles.cmYLabel}>True label</div>
                <div>
                  <div className={styles.cmGrid}>
                    {/* blank corner */}
                    <div />
                    {/* column headers */}
                    {cm.labels.map(l => (
                      <div key={l} className={styles.cmHeaderCell}>{l}</div>
                    ))}
                    {/* data rows */}
                    {cm.matrix.map((row, i) => (
                      [
                        <div key={`lbl-${i}`} className={styles.cmHeaderCell}>{cm.labels[i]}</div>,
                        ...row.map((val, j) => (
                          <div
                            key={`cell-${i}-${j}`}
                            className={styles.cmCell}
                            style={{ background: `rgba(0,229,200,${cm.normalized[i][j].toFixed(3)})` }}
                          >
                            {val}
                          </div>
                        ))
                      ]
                    ))}
                  </div>
                  <div className={styles.cmXLabel}>Predicted label</div>
                </div>
              </div>
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
