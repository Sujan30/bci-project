import { useState, useEffect } from 'react';
import { API } from './constants';
import Header        from './components/Header';
import PipelineNav   from './components/PipelineNav';
import UploadPane    from './components/UploadPane';
import PreprocessPane from './components/PreprocessPane';
import TrainPane     from './components/TrainPane';
import StreamPane    from './components/StreamPane';
import Sidebar       from './components/Sidebar';
import styles from './App.module.css';

export default function App() {
  const [health,        setHealth]        = useState('checking');
  const [step,          setStep]          = useState(0);
  const [uploadResult,  setUploadResult]  = useState(null);
  const [preprocessDir, setPreprocessDir] = useState(null);
  const [models,        setModels]        = useState([]);
  const [modLoad,  setModLoad]  = useState(false);
  const [selected, setSelected] = useState('');

  async function checkHealth() {
    try {
      const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(3000) });
      setHealth(r.ok ? 'online' : 'error');
    } catch {
      setHealth('error');
    }
  }

  async function fetchModels() {
    setModLoad(true);
    try {
      const r = await fetch(`${API}/v1/models`);
      const d = await r.json();
      setModels(d.models || []);
    } catch {
      setModels([]);
    } finally {
      setModLoad(false);
    }
  }

  useEffect(() => {
    checkHealth();
    fetchModels();
    const id = setInterval(checkHealth, 30_000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className={styles.app}>
      <Header health={health} />

      <div className={styles.main}>
        <div className={styles.pipeline}>
          <PipelineNav active={step} onChange={setStep} />

          {step === 0 && <UploadPane    onDone={(d) => { setUploadResult(d); setStep(1); }} />}
          {step === 1 && <PreprocessPane sessionId={uploadResult?.session_id} onDone={(dir) => { setPreprocessDir(dir); setStep(2); }} />}
          {step === 2 && <TrainPane     npzDir={preprocessDir} onTrained={fetchModels} />}
          {step === 3 && <StreamPane    models={models} />}
        </div>

        <Sidebar
          models={models}
          loading={modLoad}
          onRefresh={fetchModels}
          selected={selected}
          onSelect={setSelected}
        />
      </div>
    </div>
  );
}