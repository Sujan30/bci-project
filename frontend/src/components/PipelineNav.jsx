import styles from './PipelineNav.module.css';

const STEPS = [
  { id: 'upload',     label: 'Upload EDF'  },
  { id: 'preprocess', label: 'Preprocess'  },
  { id: 'train',      label: 'Train Model' },
  { id: 'stream',     label: 'Live Stream' },
];

export default function PipelineNav({ active, onChange }) {
  return (
    <nav className={styles.nav}>
      {STEPS.map((s, i) => (
        <button
          key={s.id}
          className={`${styles.btn} ${active === i ? styles.active : ''}`}
          onClick={() => onChange(i)}
        >
          <span className={styles.num}>0{i + 1}</span>
          {s.label}
        </button>
      ))}
    </nav>
  );
}
