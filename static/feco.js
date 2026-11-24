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

  const hull = document.getElementById('hull-plot');
  const fixedPhases = Array.from(hull.querySelectorAll('#fixed-phases circle'), dot => ({
    x: Number(dot.dataset.x),
    energy: Number(dot.dataset.energy)
  }));
  const b2 = fixedPhases.find(phase => phase.x === 0.5);
  const hullX = x => Number(hull.dataset.left) + x * (Number(hull.dataset.right) - Number(hull.dataset.left));
  const hullY = energy => Number(hull.dataset.zero) - energy * Number(hull.dataset.scale);

  function renderHull(delta) {
    // Only the equiatomic solid solution can lower the ordered-phase hull.
    // Build the lower convex envelope in order of increasing Co fraction.
    const bccEnergy = b2.energy + delta;
    const vertices = [];
    for (const phase of fixedPhases) {
      const point = phase === b2 && delta < 0 ? { x: b2.x, energy: bccEnergy } : phase;
      while (vertices.length >= 2) {
        const a = vertices[vertices.length - 2];
        const b = vertices[vertices.length - 1];
        if ((b.energy - a.energy) * (point.x - b.x) < (point.energy - b.energy) * (b.x - a.x)) break;
        vertices.pop();
      }
      vertices.push(point);
    }
    document.getElementById('hull-line').setAttribute('points',
      vertices.map(p => hullX(p.x) + ',' + hullY(p.energy)).join(' '));
    document.getElementById('bcc-phase').setAttribute('cy', hullY(bccEnergy));
    const distance = document.getElementById('hull-distance');
    distance.setAttribute('y1', hullY(bccEnergy));
    distance.setAttribute('y2', hullY(Math.min(b2.energy, bccEnergy)));
    document.getElementById('hull-status').textContent = delta > 0
      ? 'Disordered BCC is ' + delta.toFixed(delta < 0.1 ? 2 : 1) + ' meV/atom above the hull.'
      : 'Disordered BCC is on the hull.';
  }

  function render() {
    const t = Number(slider.value);
    const delta = enthalpy - entropy * t;
    renderHull(delta);
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
