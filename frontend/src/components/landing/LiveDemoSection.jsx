import StreamPane from '../StreamPane';
import EEGCanvas from '../EEGCanvas';
import styles from './LiveDemoSection.module.css';

export default function LiveDemoSection() {
  return (
    <section className={styles.section} id="demo">
      <EEGCanvas opacity={0.08} />
      <div className={styles.inner}>
        <div className={styles.header}>
          <span className={styles.sectionTag}>LIVE DEMO</span>
          <span className={styles.liveBadge}>
            <span className={styles.liveDot} />
            RUNNING IN BROWSER
          </span>
        </div>
        <h2 className={styles.title}>
          Watch the classifier score sleep in real time.
        </h2>
        <p className={styles.sub}>
          Demo mode runs entirely in the browser — no backend, no sign-up.
        </p>
        <div className={styles.demoFrame}>
          <StreamPane forceDemoMode={true} className={styles.streamPane} />
        </div>
        <p className={styles.note}>
          Connect your own EDF file and trained model in the{' '}
          <a href="/console" className={styles.consoleLink}>full console →</a>
        </p>
      </div>
    </section>
  );
}
