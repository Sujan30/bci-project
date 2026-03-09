import EEGCanvas from './EEGCanvas';
import styles from './Header.module.css';

export default function Header({ health }) {
  const dotCls   = health === 'online' ? 'on' : health === 'error' ? 'err' : '';
  const statusTx = health === 'online' ? 'SYSTEM ONLINE'
                 : health === 'error'  ? 'API UNREACHABLE'
                 : 'CHECKING…';

  return (
    <header className={styles.header}>
      <EEGCanvas />

      {/* CRT scanlines */}
      <div className={styles.scanlines} />

      <div className={styles.inner}>
        <div>
          <div className={styles.title}>
            SLEEP·<em>BCI</em>
          </div>
          <div className={styles.sub}>
            EEG Neural Interface Console &nbsp;·&nbsp; v1.0
          </div>
        </div>

        <div className={styles.right}>
          <div className={styles.badge}>
            <div className={`${styles.dot} ${styles[dotCls]}`} />
            {statusTx}
          </div>
          <div className={`${styles.badge} ${styles.urlBadge}`}>
            localhost:8000
          </div>
        </div>
      </div>
    </header>
  );
}
