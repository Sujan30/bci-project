import { useState } from 'react';
import CodeBlock from '../shared/CodeBlock';
import styles from './APISection.module.css';

const SNIPPETS = [
  {
    id: 'upload',
    label: 'POST /v1/upload',
    language: 'bash',
    code: `# Upload a polysomnography EDF file
curl -X POST http://localhost:8000/v1/upload \\
  -F "psg_file=@night1.edf" \\
  -F "hypnogram_file=@night1_hypno.edf"

# → { "session_id": "abc123", "channels": [...] }`,
  },
  {
    id: 'train',
    label: 'POST /v1/train',
    language: 'bash',
    code: `# Train a classifier on preprocessed epochs
curl -X POST http://localhost:8000/v1/train \\
  -H "Content-Type: application/json" \\
  -d '{
    "model_type": "random_forest",
    "dataset_id": "SC4001E0",
    "n_splits": 5
  }'

# → { "job_id": "train_xyz", "status": "queued" }`,
  },
  {
    id: 'stream',
    label: 'WS /v1/stream',
    language: 'js',
    code: `// Open WebSocket stream for real-time predictions
const ws = new WebSocket(
  'ws://localhost:8000/v1/stream?model_id=rf_baseline'
);

ws.onopen = () => {
  // Send a 30s epoch (3000 samples @ 100Hz)
  ws.send(JSON.stringify({ epoch: Float32Array }));
};

ws.onmessage = (e) => {
  const { stage, confidence, latency_ms } = JSON.parse(e.data);
  console.log(stage);      // "N2"
  console.log(latency_ms); // 1.8
};`,
  },
];

export default function APISection() {
  const [activeTab, setActiveTab] = useState('upload');
  
  const activeSnippet = SNIPPETS.find(s => s.id === activeTab);

  return (
    <section className={styles.section} id="api">
      <div className={styles.header}>
        <h2 className={styles.title}>Developer First</h2>
      </div>
      <div>
        <div className={styles.tabs}>
          {SNIPPETS.map(s => (
             <button
               key={s.id}
               className={styles.tab}
               data-active={activeTab === s.id}
               onClick={() => setActiveTab(s.id)}
             >
               {s.label}
             </button>
          ))}
        </div>
        <div className={styles.codeContainer}>
           <CodeBlock 
             code={activeSnippet.code} 
             language={activeSnippet.language} 
             showCopy={true} 
           />
        </div>
        <p className={styles.docsNote}>
          <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer">
            Explore complete OpenAPI specification &rarr;
          </a>
        </p>
      </div>
    </section>
  );
}
