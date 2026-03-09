export const API = 'http://localhost:8000';
export const WS  = 'ws://localhost:8000';

export const STAGES = {
  W:   { label: 'W',   color: '#fbbf24', desc: 'Wakefulness' },
  N1:  { label: 'N1',  color: '#34d399', desc: 'Light Sleep' },
  N2:  { label: 'N2',  color: '#60a5fa', desc: 'Core Sleep'  },
  N3:  { label: 'N3',  color: '#a78bfa', desc: 'Deep NREM'   },
  REM: { label: 'REM', color: '#f472b6', desc: 'Dreaming'    },
};

export const STAGE_ORDER = ['W', 'N1', 'N2', 'N3', 'REM'];

/* Demo simulation sequence */
export const DEMO_SEQ = ['N2','N2','N3','N3','N3','N2','REM','REM','N1','W','N1','N2','N2','N3','REM','N2'];
