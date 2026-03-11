import { useState } from 'react';
import styles from './CodeBlock.module.css';

function tokenize(code, language) {
  const tokens = [];
  const lines = code.split('\n');

  for (const line of lines) {
    if ((language === 'bash' || language === 'python') && line.trim().startsWith('#')) {
      tokens.push({ text: line + '\n', type: 'comment' });
      continue;
    }
    if (language === 'js' && line.trim().startsWith('//')) {
      tokens.push({ text: line + '\n', type: 'comment' });
      continue;
    }

    const words = line.split(/(\s+|['"].*?['"]|[{}()[\]=,;])/g).filter(Boolean);
    let lineTokens = [];

    for (const w of words) {
      if (w.match(/^\s+$/)) {
        lineTokens.push({ text: w, type: 'default' });
      } else if (w.match(/^['"].*?['"]$/)) {
        lineTokens.push({ text: w, type: 'string' });
      } else if (w.match(/^https?:\/\//) || w.match(/^ws:\/\//)) {
        lineTokens.push({ text: w, type: 'url' });
      } else if (w.match(/^-{1,2}[\w-]+$/)) {
        lineTokens.push({ text: w, type: 'flag' });
      } else if (
        ['curl', 'const', 'new', 'WebSocket', 'JSON', 'import', 'from', 'export'].includes(w)
      ) {
        lineTokens.push({ text: w, type: 'keyword' });
      } else {
        lineTokens.push({ text: w, type: 'default' });
      }
    }
    tokens.push(...lineTokens);
    tokens.push({ text: '\n', type: 'default' });
  }

  if (tokens.length && tokens[tokens.length - 1].text === '\n') {
    tokens.pop();
  }

  return tokens;
}

export default function CodeBlock({ code, language = 'bash', label, showCopy = true }) {
  const [copied, setCopied] = useState(false);

  function handleCopy() {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  const tokens = tokenize(code, language);

  return (
    <div className={styles.block}>
      {(label || showCopy) && (
        <div className={styles.header}>
          <span>{label}</span>
          {showCopy && (
            <button
              className={styles.copyBtn}
              data-copied={copied}
              onClick={handleCopy}
            >
              {copied ? 'COPIED ✓' : 'COPY'}
            </button>
          )}
        </div>
      )}
      <pre className={styles.pre}>
        <code>
          {tokens.map((t, i) => (
            <span key={i} className={styles[t.type] || styles.default}>
              {t.text}
            </span>
          ))}
        </code>
      </pre>
    </div>
  );
}
