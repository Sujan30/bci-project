import CodeBlock from '../shared/CodeBlock';
import styles from './QuickstartSection.module.css';

const STEPS = [
  {
    num: '01',
    label: 'Clone the repository',
    code: 'git clone https://github.com/Sujan30/bci-project && cd bci-project',
    language: 'bash',
  },
  {
    num: '02',
    label: 'Install and start the backend',
    code: 'pip install -e ".[dev]"\nuvicorn src.sleep_bci.api.app:app --reload',
    language: 'bash',
  },
  {
    num: '03',
    label: 'Start the frontend',
    code: 'cd frontend && npm install && npm run dev',
    language: 'bash',
  },
];

export default function QuickstartSection() {
  return (
    <section className={styles.section} id="quickstart">
      <div className={styles.header}>
        <h2 className={styles.title}>Run it locally</h2>
      </div>
      <div className={styles.stepsCol}>
        {STEPS.map(step => (
          <CodeBlock
            key={step.num}
            code={step.code}
            language={step.language}
            label={`${step.num} \u00B7 ${step.label}`}
          />
        ))}
      </div>
      <div className={styles.links}>
        <a href="https://github.com/Sujan30/bci-project" target="_blank" rel="noopener noreferrer">
          Full README on GitHub
        </a>
        <span>&middot;</span>
        <span>OpenAPI docs at localhost:8000/docs</span>
      </div>
      <p className={styles.sampleNote}>
        Sample EDF data: <code>scripts/download_sample_data.sh</code>
      </p>
    </section>
  );
}
