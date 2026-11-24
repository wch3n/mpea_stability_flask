// Formation energies in eV/atom, relative to ground-state Al and Co.
// The bundled model gives these AlCo values at 0 K. At equiatomic
// composition the HCP free energy decreases by k_B T ln(2).
(() => {
  const hcpAtZero = -0.261382889375;
  const b2Energy = -0.612345615;
  const meltingPoint = 1350.735;
  const entropy = 8.61733262e-5 * Math.log(2);
  const factor = document.getElementById('factor');
  const sweep = document.getElementById('sweep');
  const resultLink = document.getElementById('example-result');
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  let frame = 0;

  function render() {
    const fraction = Number(factor.value);
    const temperature = meltingPoint * fraction;
    const energy = hcpAtZero - temperature * entropy;
    const y = 38 - 300 * energy;
    document.getElementById('solution').setAttribute('transform', `translate(247 ${y})`);
    document.getElementById('gap-line').setAttribute('d', `M247 ${y}V216`);
    document.getElementById('fraction').value = fraction.toFixed(2);
    document.getElementById('temperature').textContent = Math.round(temperature);
    document.getElementById('gap').textContent = ((energy - b2Energy) * 1000).toFixed(1);
    resultLink.href = resultLink.dataset.url.replace('FACTOR', String(fraction));
    document.getElementById('link-factor').textContent = fraction.toFixed(2);
    factor.setAttribute('aria-valuetext', `${fraction.toFixed(2)} times the melting point, ${Math.round(temperature)} kelvin`);
  }

  function stop() {
    cancelAnimationFrame(frame);
    frame = 0;
    sweep.textContent = 'Animate heating';
  }

  sweep.addEventListener('click', () => {
    if (frame) { stop(); return; }
    const start = performance.now();
    sweep.textContent = 'Pause heating';
    function advance(time) {
      const value = Math.min(1, (time - start) / 3800);
      factor.value = value;
      render();
      if (value < 1) frame = requestAnimationFrame(advance);
      else stop();
    }
    frame = requestAnimationFrame(advance);
  });
  factor.addEventListener('input', () => { stop(); render(); });
  motion.addEventListener('change', () => { stop(); sweep.hidden = motion.matches; });
  sweep.hidden = motion.matches;
  render();
  document.getElementById('controls').hidden = false;
})();
