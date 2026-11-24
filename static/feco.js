(() => {
  const plot = document.getElementById('phase-plot');
  const { enthalpy, entropy, melting, left, right, zero, scale, maxT } =
    Object.fromEntries(Object.entries(plot.dataset).map(([key, value]) => [key, Number(value)]));
  const slider = document.getElementById('temperature');
  const marker = document.getElementById('cursor');
  const separation = document.getElementById('separation');
  const result = document.getElementById('example-result');
  const sweep = document.getElementById('sweep');
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  let frame = 0;

  function render() {
    const t = Number(slider.value);
    const delta = enthalpy - entropy * t;
    const x = left + (right - left) * t / maxT;
    const y = zero - delta * scale;
    marker.setAttribute('cx', x);
    marker.setAttribute('cy', y);
    separation.setAttribute('x1', x);
    separation.setAttribute('x2', x);
    separation.setAttribute('y1', y);
    document.getElementById('kelvin').value = t + ' K';
    document.getElementById('winner').textContent = delta > 0 ? 'Ordered B2' : 'Disordered BCC';
    document.getElementById('advantage').textContent =
      Math.abs(delta).toFixed(Math.abs(delta) < 0.1 ? 2 : 1);
    document.getElementById('entropy-credit').textContent = (entropy * t).toFixed(1);
    slider.setAttribute('aria-valuetext', t + ' kelvin');
    result.href = result.dataset.url.replace('FACTOR', (t / melting).toFixed(9));
  }

  function pause() {
    cancelAnimationFrame(frame);
    frame = 0;
    sweep.textContent = Number(slider.value) === maxT ? 'Replay heating' : 'Animate heating';
  }

  sweep.addEventListener('click', () => {
    if (frame) { pause(); return; }
    const initial = Number(slider.value) === maxT ? 0 : Number(slider.value);
    const start = performance.now();
    sweep.textContent = 'Pause heating';
    function advance(now) {
      const t = Math.min(maxT, initial + Math.max(0, now - start) * maxT / 6500);
      slider.value = Math.round(t);
      render();
      if (t < maxT) frame = requestAnimationFrame(advance);
      else pause();
    }
    frame = requestAnimationFrame(advance);
  });
  slider.addEventListener('input', () => { pause(); render(); });
  motion.addEventListener('change', () => { pause(); sweep.hidden = motion.matches; });
  sweep.hidden = motion.matches;
  render();
  document.getElementById('controls').hidden = false;
})();
