import { useEffect, useRef } from 'react';
import styles from './PipelineDiagram.module.css';

const NODES = [
  {
    num: '01',
    title: 'EDF Upload',
    detail: 'EDF / EDF+ native\nMNE validates channels\nsampling rate check',
    edgeLabel: 'raw signal',
  },
  {
    num: '02',
    title: 'Preprocess',
    detail: 'bandpass 0.5–30 Hz\n30s epoch windows\nWelch PSD features',
    edgeLabel: 'feature matrix',
  },
  {
    num: '03',
    title: 'Train Classifier',
    detail: 'LDA or Random Forest\nscikit-learn fit\nserialised to disk',
    edgeLabel: 'model bundle',
  },
  {
    num: '04',
    title: 'Stream Inference',
    detail: 'WebSocket push\n<10ms per epoch\nhypnogram render',
    edgeLabel: null,
  },
];

export default function PipelineDiagram() {
  const diagramRef = useRef(null);
  useEffect(() => {
    const el = diagramRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) el.classList.add(styles.visible); },
      { threshold: 0.2 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <section className={styles.section} id="pipeline">
      <div className={styles.diagram} ref={diagramRef}>
        {NODES.map((node, i) => (
          <div key={node.num} style={{ display: 'contents' }}>
            <div className={styles.node} style={{ '--node-i': i }}>
              <div className={styles.nodeNum}>{node.num}</div>
              <h3 className={styles.nodeTitle}>{node.title}</h3>
              <p className={styles.nodeDetail}>{node.detail}</p>
            </div>
            {node.edgeLabel && (
              <div className={styles.connectorCol}>
                <span className={styles.edgeLabel}>{node.edgeLabel}</span>
                <svg width="48" height="2" className={styles.connectorLine}>
                  <line x1="0" y1="1" x2="48" y2="1" stroke="var(--cyan)" strokeWidth="2" />
                </svg>
              </div>
            )}
          </div>
        ))}
      </div>
      <div className={styles.routeMap}>
        POST /v1/upload  &rarr;  POST /v1/preprocess  &rarr;  POST /v1/train  &rarr;  WS /v1/stream
      </div>
    </section>
  );
}
