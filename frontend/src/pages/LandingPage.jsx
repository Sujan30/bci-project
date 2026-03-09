import { Link } from 'react-router-dom';
import EEGCanvas from '../components/EEGCanvas';
import styles from './LandingPage.module.css';

/* ── Icons ────────────────────────────────────────────────────────────── */

// Lucide-style: 24×24 viewBox, stroke="currentColor", strokeWidth=1.5,
// strokeLinecap/Join="round", fill="none"

function IconWaveform() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.5"
      strokeLinecap="round" strokeLinejoin="round">
      {/* Flat → EEG spike → flat — mirrors the EEGCanvas wave aesthetic */}
      <polyline points="2,12 5,12 6.5,7 8,17 10,4 12,20 14,9 16,15 17.5,12 22,12" />
    </svg>
  );
}

function IconSpectrum() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.5"
      strokeLinecap="round">
      {/* 5 frequency bars: δ θ α β γ — each a different height */}
      <line x1="3"  y1="20" x2="3"  y2="15" />
      <line x1="7"  y1="20" x2="7"  y2="8"  />
      <line x1="11" y1="20" x2="11" y2="4"  />
      <line x1="15" y1="20" x2="15" y2="10" />
      <line x1="19" y1="20" x2="19" y2="16" />
      {/* Smoothed spectral envelope over the bars */}
      <path d="M2,15 C4,15 5,8 7,8 C9,8 9,4 11,4 C13,4 13,10 15,10 C17,10 18,16 20,16" />
    </svg>
  );
}

function IconFileSignal() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.5"
      strokeLinecap="round" strokeLinejoin="round">
      {/* Document body with folded top-right corner */}
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14,2 14,8 20,8" />
      {/* Mini waveform embedded inside the document */}
      <polyline points="7,15 8.5,15 9.5,12.5 10.5,17.5 11.5,11 12.5,19 13.5,14 15,14 16,15 17,15" />
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
];

const STEPS = [
  { num: '01', title: 'Upload EDF',       desc: 'Drop a raw EDF/EDF+ polysomnography file. The API validates channels and sampling rate.' },
  { num: '02', title: 'Preprocess',       desc: 'MNE-Python filters the signal, extracts 30-second epochs, and computes spectral features.' },
  { num: '03', title: 'Train Classifier', desc: 'scikit-learn trains a Random Forest on the feature matrix. Model is serialized and stored server-side.' },
  { num: '04', title: 'Stream Results',   desc: 'WebSocket delivers real-time stage predictions as a hypnogram. Latency under 10ms per epoch.' },
];

const FEATURES = [
  { Icon: IconWaveform,    title: 'Real-Time WebSocket Stream',    desc: 'Persistent bidirectional socket pushes epoch predictions as they are scored. No polling, no delay.' },
  { Icon: IconSpectrum,    title: 'Spectral Feature Extraction',   desc: 'Band-power ratios (δ, θ, α, β) computed per epoch via Welch PSD. Features fed directly to the classifier.' },
  { Icon: IconFileSignal,  title: 'EDF/EDF+ Native Support',       desc: 'Upload standard polysomnography files directly. No conversion step. MNE handles channel selection automatically.' },
];

const TECH = [
  { label: 'MNE-Python',   color: '#60a5fa' },
  { label: 'scikit-learn', color: '#34d399' },
  { label: 'FastAPI',      color: '#a78bfa' },
  { label: 'WebSocket',    color: '#00e5c8' },
  { label: 'EDF/EDF+',     color: '#fbbf24' },
  { label: 'React 19',     color: '#f472b6' },
];

/* ── Sub-components ───────────────────────────────────────────────────── */

function Navbar() {
  return (
    <nav className={styles.nav}>
      <div className={styles.navLogo}>
        NEURAL<em>SLEEP</em>
      </div>
      <Link to="/app" className={styles.navCta}>
        Launch Console →
      </Link>
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
          <Link to="/app" className={styles.btnPrimary}>
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

function HowItWorks() {
  return (
    <div className={styles.section}>
      <div className={styles.sectionTag}>Pipeline</div>
      <h2 className={styles.sectionTitle}>Four steps, zero guesswork.</h2>
      <div className={styles.stepsGrid}>
        {STEPS.map(s => (
          <div key={s.num} className={styles.stepCard}>
            <div className={styles.stepNum}>{s.num}</div>
            <div className={styles.stepTitle}>{s.title}</div>
            <p className={styles.stepDesc}>{s.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function FeatureCards() {
  return (
    <div className={styles.section}>
      <div className={styles.sectionTag}>Features</div>
      <h2 className={styles.sectionTitle}>Built for the real pipeline.</h2>
      <div className={styles.featGrid}>
        {FEATURES.map(({ Icon, title, desc }) => (
          <div key={title} className={styles.featCard}>
            <div className={styles.featIcon}>
              <Icon />
            </div>
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
      <h2 className={styles.sectionTitle}>Production-grade components.</h2>
      <div className={styles.techBadges}>
        {TECH.map(t => (
          <span
            key={t.label}
            className={styles.techBadge}
            style={{ '--badge-color': t.color }}
          >
            {t.label}
          </span>
        ))}
      </div>
    </div>
  );
}

function Footer() {
  return (
    <footer>
      <div className={styles.footer}>
        <div className={styles.footerLogo}>
          NEURAL<em>SLEEP</em> &nbsp;·&nbsp; EEG Sleep Stage Classification
        </div>
        <div className={styles.footerLinks}>
          <a
            href="https://github.com/Sujan30/bci-project"
            target="_blank"
            rel="noopener noreferrer"
            className={styles.footerLink}
          >
            GitHub
          </a>
          <Link to="/app" className={styles.footerLink}>
            Console
          </Link>
        </div>
      </div>
    </footer>
  );
}

/* ── Page ─────────────────────────────────────────────────────────────── */

export default function LandingPage() {
  return (
    <>
      <Navbar />
      <Hero />
      <StatsBar />
      <ProblemSolution />
      <HowItWorks />
      <FeatureCards />
      <TechStack />
      <Footer />
    </>
  );
}
