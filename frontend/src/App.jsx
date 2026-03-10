import { useState, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { API } from './constants';
import Header        from './components/Header';
import PipelineNav   from './components/PipelineNav';
import UploadPane    from './components/UploadPane';
import PreprocessPane from './components/PreprocessPane';
import TrainPane     from './components/TrainPane';
import StreamPane    from './components/StreamPane';
import Sidebar       from './components/Sidebar';
import LandingPage   from './pages/LandingPage';
import styles from './App.module.css';

function PipelineApp() {
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

          <div style={{ display: step === 0 ? 'block' : 'none' }}>
            <UploadPane onDone={(d) => { setUploadResult(d); setStep(1); }} />
          </div>
          <div style={{ display: step === 1 ? 'block' : 'none' }}>
            <PreprocessPane sessionId={uploadResult?.session_id} onDone={(dir) => { setPreprocessDir(dir); setStep(2); }} />
          </div>
          <div style={{ display: step === 2 ? 'block' : 'none' }}>
            <TrainPane npzDir={preprocessDir} onTrained={fetchModels} />
          </div>
          <div style={{ display: step === 3 ? 'block' : 'none' }}>
            <StreamPane models={models} />
          </div>
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

export default function App() {
  return (
    <Routes>
      <Route path="/"    element={<LandingPage />} />
      <Route path="/app" element={<PipelineApp />} />
    </Routes>
  );
}