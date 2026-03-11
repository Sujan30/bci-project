import { useState, useEffect } from 'react';
import styles from './ModelMetricsSection.module.css';
import { STAGES } from '../../constants';

const PLACEHOLDER = {
  lda: {
    balanced_accuracy: 0.76,
    macro_f1: 0.69,
    f1_per_stage: { W: 0.90, N1: 0.38, N2: 0.80, N3: 0.76, REM: 0.72 },
    confusion_matrix: {
      labels: ['W','N1','N2','N3','REM'],
      matrix: [
        [142, 10,  3,  0,  5],
        [ 14, 35, 20,  2,  7],
        [  4, 16,193, 11,  6],
        [  0,  2, 13, 84,  1],
        [  7, 10,  9,  0, 89],
      ],
      normalized: [
        [0.89, 0.06, 0.02, 0, 0.03],
        [0.18, 0.45, 0.26, 0.03, 0.09],
        [0.02, 0.07, 0.84, 0.05, 0.03],
        [0, 0.02, 0.13, 0.84, 0.01],
        [0.06, 0.09, 0.08, 0, 0.77],
      ],
    },
  },
  rf: {
    balanced_accuracy: 0.83,
    macro_f1: 0.76,
    f1_per_stage: { W: 0.93, N1: 0.52, N2: 0.86, N3: 0.83, REM: 0.81 },
    confusion_matrix: {
      labels: ['W','N1','N2','N3','REM'],
      matrix: [
        [150,  4,  3,  0,  3],
        [  9, 50, 15,  1,  3],
        [  2,  9,204,  7,  8],
        [  0,  1,  8, 90,  1],
        [  4,  6,  7,  0, 98],
      ],
      normalized: [
        [0.94, 0.03, 0.02, 0, 0.02],
        [0.12, 0.64, 0.19, 0.01, 0.04],
        [0.01, 0.04, 0.89, 0.03, 0.03],
        [0, 0.01, 0.08, 0.90, 0.01],
        [0.03, 0.05, 0.06, 0, 0.86],
      ],
    },
  },
};

const STAGE_COLORS = { W: '#fbbf24', N1: '#34d399', N2: '#60a5fa', N3: '#a78bfa', REM: '#f472b6' };

export default function ModelMetricsSection() {
  const [metricsData, setMetricsData] = useState(null);
  const [activeModel, setActiveModel] = useState('rf');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const ctrl = new AbortController();
    (async () => {
      try {
        const r = await fetch('/v1/models', { signal: ctrl.signal });
        if (!r.ok) throw new Error();
        const { models } = await r.json();
        if (!models?.length) throw new Error();
        const m = await fetch(`/v1/models/${models[0].id}/metrics`, { signal: ctrl.signal });
        if (!m.ok) throw new Error();
        setMetricsData(await m.json());
      } catch { 
        setMetricsData(PLACEHOLDER);
      } finally {
        setLoading(false);
      }
    })();
    return () => ctrl.abort();
  }, []);

  if (loading) return <div className={styles.section}>Loading metrics...</div>;

  const data = metricsData[activeModel];
  if (!data) return null;

  const isPlaceholder = metricsData === PLACEHOLDER;

  return (
    <section className={styles.section} id="metrics">
      <div className={styles.header}>
        <div className={styles.titleRow}>
          <h2 className={styles.title}>Model Accuracy</h2>
          <div className={styles.modelToggle}>
            <button 
              className={`${styles.toggleBtn} ${activeModel === 'lda' ? styles.toggleActive : ''}`}
              onClick={() => setActiveModel('lda')}
            >
              LDA
            </button>
            <button 
              className={`${styles.toggleBtn} ${activeModel === 'rf' ? styles.toggleActive : ''}`}
              onClick={() => setActiveModel('rf')}
            >
              Random Forest
            </button>
          </div>
        </div>
        <p className={styles.subtitle}>
          Balanced Accuracy: {(data.balanced_accuracy * 100).toFixed(1)}% &middot; Macro F1: {(data.macro_f1 * 100).toFixed(1)}%
        </p>
      </div>

      <div className={styles.grid}>
        {/* F1 Bars */}
        <div className={styles.f1Container}>
          <h3 className={styles.panelTitle}>F1 Score per Stage</h3>
          <div className={styles.f1Bars}>
           {data.confusion_matrix.labels.map((stage, i) => (
             <div key={stage} className={styles.barRow}>
               <span className={styles.barLabel}>{stage}</span>
               <div className={styles.barTrack}>
                 <div 
                   className={styles.barFill} 
                   style={{ 
                     '--bar-width': `${data.f1_per_stage[stage] * 100}%`,
                     '--bar-delay': `${i * 100}ms`,
                     background: STAGE_COLORS[stage] 
                   }} 
                 />
               </div>
               <span className={styles.barVal}>{(data.f1_per_stage[stage] * 100).toFixed(0)}</span>
             </div>
           ))}
          </div>
        </div>

        {/* Confusion Matrix */}
        <div className={styles.matrixContainer}>
          <h3 className={styles.panelTitle}>Confusion Matrix (Normalized)</h3>
          <div className={styles.matrixWrapper}>
            <div className={styles.matrix}>
               {/* Header Row */}
               <div className={styles.matrixHeaderCorner} />
               {data.confusion_matrix.labels.map(lbl => (
                 <div key={lbl} className={styles.matrixHeaderX}>{lbl}</div>
               ))}
               
               {/* Data Rows */}
               {data.confusion_matrix.matrix.map((row, rIdx) => (
                 <div key={rIdx} style={{ display: 'contents' }}>
                   <div className={styles.matrixHeaderY}>{data.confusion_matrix.labels[rIdx]}</div>
                   {row.map((val, cIdx) => (
                     <div 
                       key={cIdx} 
                       className={styles.matrixCell}
                       data-diagonal={rIdx === cIdx}
                       style={{ background: `rgba(0,229,200, ${data.confusion_matrix.normalized[rIdx][cIdx]})` }}
                       title={`True: ${data.confusion_matrix.labels[rIdx]}, Pred: ${data.confusion_matrix.labels[cIdx]} \n Count: ${val}`}
                     >
                        {(data.confusion_matrix.normalized[rIdx][cIdx] * 100).toFixed(0)}
                     </div>
                   ))}
                 </div>
               ))}
            </div>
          </div>
        </div>
      </div>

      {isPlaceholder && (
         <p className={styles.footerNote}>
           * Placeholder data. Connect a trained model to see real metrics.
         </p>
      )}
    </section>
  );
}
