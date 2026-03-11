import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import EEGCanvas from '../components/EEGCanvas';
import LiveDemoSection from '../components/landing/LiveDemoSection';
import PipelineDiagram from '../components/landing/PipelineDiagram';
import APISection from '../components/landing/APISection';
import ModelMetricsSection from '../components/landing/ModelMetricsSection';
import QuickstartSection from '../components/landing/QuickstartSection';
import styles from './LandingPage.module.css';

/* ── Icons ────────────────────────────────────────────────────────────── */

function IconWaveform() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.5"
      strokeLinecap="round" strokeLinejoin="round">
      <polyline points="2,12 5,12 6.5,7 8,17 10,4 12,20 14,9 16,15 17.5,12 22,12" />
    </svg>
  );
}

function IconSpectrum() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.5"
      strokeLinecap="round">
      <line x1="3"  y1="20" x2="3"  y2="15" />
      <line x1="7"  y1="20" x2="7"  y2="8"  />
      <line x1="11" y1="20" x2="11" y2="4"  />
      <line x1="15" y1="20" x2="15" y2="10" />
      <line x1="19" y1="20" x2="19" y2="16" />
      <path d="M2,15 C4,15 5,8 7,8 C9,8 9,4 11,4 C13,4 13,10 15,10 C17,10 18,16 20,16" />
    </svg>
  );
}

function IconFileSignal() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.5"
      strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14,2 14,8 20,8" />
      <polyline points="7,15 8.5,15 9.5,12.5 10.5,17.5 11.5,11 12.5,19 13.5,14 15,14 16,15 17,15" />
    </svg>
  );
}

function IconClassifier() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 20 L12 8 L21 20" />
      <circle cx="12" cy="8" r="2" />
      <line x1="3" y1="14" x2="21" y2="14" strokeDasharray="3 2" />
    </svg>
  );
}

function IconHypnogram() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="2,8 6,8 6,12 10,12 10,18 14,18 14,12 18,12 18,6 22,6" />
    </svg>
  );
}

/* ── Data ─────────────────────────────────────────────────────────────── */

const STAGES = [
  { label: 'Wake', color: 'var(--wake)' },
  { label: 'N1',   color: 'var(--n1)'   },
  { label: 'N2',   color: 'var(--n2)'   },
  { label: 'N3',   color: 'var(--n3)'   },
  { label: 'REM',  color: 'var(--rem)'  },
];

const STATS = [
  { value: '<10ms',  label: 'Inference latency' },
  { value: '5',      label: 'Sleep stages'       },
  { value: '30s',    label: 'Epoch resolution'   },
  { value: '<2 min', label: 'Full pipeline'      },
  { value: '~83%',   label: 'RF accuracy*'       }, 
];

const FEATURES = [
  { Icon: IconWaveform,    title: 'Real-Time WebSocket Stream',    desc: 'Persistent bidirectional socket pushes epoch predictions as they are scored. No polling, no delay.' },
  { Icon: IconSpectrum,    title: 'Spectral Feature Extraction',   desc: 'Band-power ratios (δ, θ, α, β) computed per epoch via Welch PSD. Features fed directly to the classifier.' },
  { Icon: IconFileSignal,  title: 'EDF/EDF+ Native Support',       desc: 'Upload standard polysomnography files directly. No conversion step. MNE handles channel selection automatically.' },
  { Icon: IconClassifier,  title: 'LDA + Random Forest',           desc: 'Train on the same spectral feature matrix with either model. LDA is fast and interpretable; Random Forest captures non-linear boundaries. Select via API parameter at training time.' },
  { Icon: IconHypnogram,   title: 'Live Hypnogram Rendering',      desc: 'Every epoch prediction appends to a color-coded stage timeline. Wake, N1, N2, N3, and REM render as a live hypnogram — a real-time artifact of the scoring session, not a post-hoc export.' },
];

const TECH = [
  {
    label: 'MNE-Python',
    color: '#60a5fa',
    role: 'EEG Signal Processing',
    why: 'Industry-standard Python library for M/EEG — handles EDF parsing, bandpass filtering, and epoch extraction natively.',
  },
  {
    label: 'scikit-learn',
    color: '#34d399',
    role: 'Classification',
    why: 'Provides RandomForestClassifier and LinearDiscriminantAnalysis with a consistent fit/predict API and built-in cross-validation.',
  },
  {
    label: 'FastAPI',
    color: '#a78bfa',
    role: 'REST API + Async Runtime',
    why: 'ASGI framework with automatic OpenAPI schema generation. Background job support via ARQ + Redis, falls back to BackgroundTasks when offline.',
  },
  {
    label: 'WebSocket',
    color: '#00e5c8',
    role: 'Real-Time Transport',
    why: 'Avoids HTTP polling overhead. Epoch predictions arrive at the client as they are scored — persistent bidirectional connection.',
  },
  {
    label: 'EDF / EDF+',
    color: '#fbbf24',
    role: 'Data Format',
    why: 'International standard for polysomnography recordings. MNE reads it natively — no conversion step required before processing.',
  },
  {
    label: 'React 19',
    color: '#f472b6',
    role: 'Frontend',
    why: 'Used for component-level streaming state (StreamPane) and CSS Module scoping. No UI library dependency — styled from scratch.',
  },
];

/* ── Sub-components ───────────────────────────────────────────────────── */

function Navbar({ activeSection }) {
  const NAV_LINKS = [
    { label: 'Demo',       href: '#demo'      },
    { label: 'Pipeline',   href: '#pipeline'  },
    { label: 'API',        href: '#api'       },
    { label: 'Quickstart', href: '#quickstart'},
  ];

  return (
    <nav className={styles.nav}>
      <div className={styles.navLogo}>NEURAL<em>SLEEP</em></div>
      <div className={styles.navLinks}>
        {NAV_LINKS.map(l => (
          <a
            key={l.href}
            href={l.href}
            className={`${styles.navLink} ${activeSection === l.href.slice(1) ? styles.navLinkActive : ''}`}
          >
            {l.label}
          </a>
        ))}
      </div>
      <Link to="/console" className={styles.navCta}>Launch Console →</Link>
    </nav>
  );
}

function Hero() {
  return (
    <section className={styles.hero}>
      <EEGCanvas />
      <div className={styles.heroOverlay} />
      <div className={styles.heroContent}>
        <div className={styles.eyebrow}>EEG Sleep Stage Classification</div>
        <h1 className={styles.heroTitle}>
          Automated PSG Scoring<br />
          at <em>Neural Speed</em>
        </h1>
        <p className={styles.heroSub}>
          NeuralSleep replaces 2–4 hours of manual polysomnography annotation with a
          sub-10ms inference pipeline — upload an EDF, train, stream results live.
        </p>
        <div className={styles.heroCtas}>
          <Link to="/console" className={styles.btnPrimary}>
            Launch Console →
          </Link>
          <a
            href="https://github.com/Sujan30/bci-project"
            target="_blank"
            rel="noopener noreferrer"
            className={styles.btnGhost}
          >
            View Source
          </a>
        </div>
        <div className={styles.stagePills}>
          {STAGES.map(s => (
            <span
              key={s.label}
              className={styles.pill}
              style={{ '--pill-color': s.color }}
            >
              {s.label}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}

function StatsBar() {
  return (
    <div className={styles.statsBar}>
      {STATS.map(s => (
        <div key={s.label} className={styles.statItem}>
          <div className={styles.statValue}>{s.value}</div>
          <div className={styles.statLabel}>{s.label}</div>
        </div>
      ))}
    </div>
  );
}

function ProblemSolution() {
  return (
    <div className={styles.section}>
      <div className={styles.sectionTag}>The Problem</div>
      <h2 className={styles.sectionTitle}>Manual scoring doesn't scale.</h2>
      <div className={styles.probSolGrid}>
        <div className={styles.probCard}>
          <span className={`${styles.cardTag} ${styles.probTag}`}>Before</span>
          <div className={styles.cardTitle}>2–4 hours per patient, per night</div>
          <p className={styles.cardBody}>
            A trained technologist visually inspects 30-second epochs across
            multiple EEG, EOG, and EMG channels — assigning Wake, N1, N2, N3,
            or REM to each. One study = ~900 epochs to review. Inter-rater
            variability adds another layer of inconsistency.
          </p>
        </div>
        <div className={styles.solCard}>
          <span className={`${styles.cardTag} ${styles.solTag}`}>After</span>
          <div className={styles.cardTitle}>Under 2 minutes, end-to-end</div>
          <p className={styles.cardBody}>
            NeuralSleep preprocesses the raw signal, extracts spectral features,
            trains a classifier, and streams epoch-by-epoch predictions over
            WebSocket. The entire pipeline from EDF upload to live hypnogram
            runs in under two minutes on standard hardware.
          </p>
        </div>
      </div>
    </div>
  );
}

function FeatureCards() {
  const first = FEATURES.slice(0, 3);
  const last  = FEATURES.slice(3);
  return (
    <div className={styles.section}>
      <div className={styles.sectionTag}>Features</div>
      <h2 className={styles.sectionTitle}>Built for the real pipeline.</h2>
      <div className={styles.featGrid}>
        {first.map(({ title, desc }) => (
          <div key={title} className={styles.featCard}>
            <div className={styles.featIcon}></div>
            <div className={styles.featTitle}>{title}</div>
            <p className={styles.featDesc}>{desc}</p>
          </div>
        ))}
      </div>
      <div className={styles.featGridTwo}>
        {last.map(({ title, desc }) => (
          <div key={title} className={styles.featCard}>
            <div className={styles.featIcon}></div>
            <div className={styles.featTitle}>{title}</div>
            <p className={styles.featDesc}>{desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function TechStack() {
  return (
    <div className={styles.section}>
      <div className={styles.sectionTag}>Stack</div>
      <h2 className={styles.sectionTitle}>Every tool chosen for a reason.</h2>
      <div className={styles.techGrid}>
        {TECH.map(t => (
          <div key={t.label} className={styles.techEntry}>
            <div>
              <span className={styles.techBadge} style={{ '--badge-color': t.color }}>{t.label}</span>
            </div>
            <span className={styles.techRole}>{t.role}</span>
            <p className={styles.techWhy}>{t.why}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function Footer() {
  const NAV_LINKS = ['Demo', 'Pipeline', 'API', 'Quickstart'];
  return (
    <footer>
      <div className={styles.footerGrid}>
        <div className={styles.footerBrand}>
          <div className={styles.footerLogo}>NEURAL<em>SLEEP</em></div>
          <div className={styles.footerTagline}>EEG Sleep Stage Classification</div>
          <p className={styles.footerNote}>
            * RF accuracy: placeholder data trained on Sleep-EDF-20 dataset,
            Fpz-Cz channel, 5-class AASM scoring. Connect a real model to see live metrics.
          </p>
        </div>
        <div className={styles.footerNav}>
          {NAV_LINKS.map(l => (
            <a key={l} href={`#${l.toLowerCase()}`} className={styles.footerLink}>{l}</a>
          ))}
          <Link to="/console" className={styles.footerLink}>Console</Link>
        </div>
        <div className={styles.footerExt}>
          <a href="https://github.com/Sujan30/bci-project" target="_blank" rel="noopener noreferrer" className={styles.footerLink}>GitHub</a>
          <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer" className={styles.footerLink}>API Docs</a>
          <a href="https://github.com/Sujan30/bci-project/issues" target="_blank" rel="noopener noreferrer" className={styles.footerLink}>Issues</a>
        </div>
      </div>
      <div className={styles.footerBottom}>
        Built with MNE-Python · FastAPI · React 19
      </div>
    </footer>
  );
}

/* ── Page ─────────────────────────────────────────────────────────────── */

export default function LandingPage() {
  const [activeSection, setActiveSection] = useState(null);

  useEffect(() => {
    const sections = document.querySelectorAll('[data-section]');
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(e => {
          if (e.isIntersecting) setActiveSection(e.target.dataset.section);
        });
      },
      { threshold: 0.3 }
    );
    sections.forEach(s => observer.observe(s));
    return () => observer.disconnect();
  }, []);

  return (
    <>
      <Navbar activeSection={activeSection} />
      <Hero />
      <StatsBar />
      <div data-section="demo"><LiveDemoSection /></div>
      <div data-section="pipeline"><PipelineDiagram /></div>
      <ProblemSolution />
      <FeatureCards />
      <div data-section="api"><APISection /></div>
      <ModelMetricsSection />
      <TechStack />
      <div data-section="quickstart"><QuickstartSection /></div>
      <Footer />
    </>
  );
}
