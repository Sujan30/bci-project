import { useEffect, useRef } from 'react';

const WAVES = [
  { freq: 0.6,  amp: 12, speed: 0.0035, color: '#00e5c8', lw: 1.5 },
  { freq: 2.2,  amp: 7,  speed: 0.006,  color: '#60a5fa', lw: 1.0 },
  { freq: 4.5,  amp: 5,  speed: 0.009,  color: '#a78bfa', lw: 0.75 },
  { freq: 9.0,  amp: 3,  speed: 0.013,  color: '#f472b6', lw: 0.55 },
];

export default function EEGCanvas() {
  const canvasRef = useRef(null);
  const rafRef    = useRef(null);
  const tRef      = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    const resize = () => {
      canvas.width  = canvas.offsetWidth  * dpr;
      canvas.height = canvas.offsetHeight * dpr;
      ctx.scale(dpr, dpr);
    };

    const draw = () => {
      const W = canvas.offsetWidth;
      const H = canvas.offsetHeight;
      ctx.clearRect(0, 0, W, H);
      tRef.current += 0.9;
      const mid = H / 2;

      WAVES.forEach(w => {
        ctx.beginPath();
        ctx.strokeStyle = w.color;
        ctx.lineWidth   = w.lw;
        ctx.globalAlpha = 0.55;

        for (let x = 0; x <= W; x += 1.5) {
          const ph = tRef.current * w.speed;
          const y  = mid
            + Math.sin((x / W) * Math.PI * 2 * w.freq + ph) * w.amp
            + Math.sin((x / W) * Math.PI * 4 * w.freq * 0.6 + ph * 1.4) * (w.amp * 0.28);
          x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
      });

      rafRef.current = requestAnimationFrame(draw);
    };

    resize();
    window.addEventListener('resize', resize);
    draw();

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        opacity: 0.32,
        display: 'block',
      }}
    />
  );
}
