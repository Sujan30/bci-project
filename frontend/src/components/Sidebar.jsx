import { STAGES, STAGE_ORDER } from '../constants';
import s from './shared.module.css';
import styles from './Sidebar.module.css';

const ENDPOINTS = [
  ['POST', '/v1/upload'],
  ['POST', '/v1/preprocess'],
  ['GET',  '/v1/preprocess/{id}'],
  ['POST', '/v1/train'],
  ['GET',  '/v1/train/{id}'],
  ['GET',  '/v1/models'],
  ['WS',   '/v1/stream'],
];

const METHOD_STYLE = {
  GET:  { bg: 'rgba(52,211,153,.1)',  color: '#34d399' },
  POST: { bg: 'rgba(96,165,250,.1)',  color: '#60a5fa' },
  WS:   { bg: 'rgba(167,139,250,.1)', color: '#a78bfa' },
};

export default function Sidebar({ models, loading, onRefresh, selected, onSelect }) {
  return (
    <aside className={styles.sidebar}>
      {/* Models */}
      <div className={`${s.card} ${styles.sticky}`}>
        <div className={s.cardTitle}>
          <span>Trained Models</span>
          <button
            className={styles.iconBtn}
            onClick={onRefresh}
            title="Refresh models"
          >
            {loading
              ? <span className={`${s.spinner} ${styles.tinySpinner}`} />
              : '↻'}
          </button>
        </div>

        {models.length === 0 ? (
          <div className={styles.empty}>No models found.</div>
        ) : models.map(m => (
          <div
            key={m}
            className={`${styles.modelItem} ${selected === m ? styles.sel : ''}`}
            onClick={() => onSelect(m)}
          >
            <div className={`${styles.mdot} ${selected === m ? styles.mdotSel : ''}`} />
            <div className={styles.modelName}>{m}</div>
          </div>
        ))}
      </div>

      {/* Sleep stages legend */}
      <div className={s.card} style={{ marginTop: 12 }}>
        <div className={s.cardTitle}>Sleep Stages</div>
        {STAGE_ORDER.map(k => (
          <div key={k} className={styles.stageRow}>
            <div
              className={styles.stageSwatch}
              style={{ background: STAGES[k].color, boxShadow: `0 0 5px ${STAGES[k].color}` }}
            />
            <div className={styles.stageKey} style={{ color: STAGES[k].color }}>{k}</div>
            <div className={styles.stageDesc}>{STAGES[k].desc}</div>
          </div>
        ))}
      </div>

      {/* API reference */}
      <div className={s.card} style={{ marginTop: 12 }}>
        <div className={s.cardTitle}>API Reference</div>
        {ENDPOINTS.map(([method, path]) => (
          <div key={path} className={styles.endpointRow}>
            <span
              className={styles.methodTag}
              style={METHOD_STYLE[method]}
            >
              {method}
            </span>
            <span className={styles.endpointPath}>{path}</span>
          </div>
        ))}
      </div>
    </aside>
  );
}