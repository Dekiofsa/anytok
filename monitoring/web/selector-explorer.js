'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const names = {uniform: 'Uniform sampling', reinforce: 'Old REINFORCE', predictor: 'New hardness predictor'};
  const methodColors = {uniform: '#87919D', reinforce: '#2F8FD0', predictor: '#36967B'};
  const colors = {bpe: '#2F8FD0', wordpiece: '#36967B', unigram: '#8657BC'};
  const methods = ['uniform', 'reinforce', 'predictor'];
  const rules = {
    fixed: ['Hardness stays fixed', 'All 32 measured rewards per tokenizer stay available. Each step samples one of those batches. Only the selector changes what it believes and chooses.'],
    gradual: ['The hard family gradually becomes easier', 'At step 10,000, every Unigram tokenizer starts getting easier. Over 1,500 steps, its mean gap falls by 0.012487 BPB—the size of #771’s observed 6.5k → 8k improvement. Other means stay fixed.'],
    abrupt: ['The hard family suddenly becomes easier', 'At step 10,000, every Unigram mean falls by 0.024974 BPB, twice the observed change. The previously hardest family is no longer the hardest. Can the selector switch?'],
    practice_tokenizer: ['Choosing a tokenizer makes that tokenizer easier', 'After each choice, that tokenizer’s mean drops a little. Other tokenizers do not improve. Picking the same one repeatedly can leave harder choices elsewhere.'],
    practice_family: ['Choosing a tokenizer makes its whole family easier', 'After each choice, every tokenizer in its algorithm family becomes a little easier. BPE, WordPiece and Unigram each have a shared practice count. The hard family can keep changing.']
  };
  let meta = null, current = null, timer = null, request = null, version = 0, selected = 771;
  let pendingFrame = null, tableSignature = '';
  const params = new URLSearchParams(location.search);
  const initialScenario = params.get('scenario') || 'practice_family';
  const initialSeed = Number(params.get('seed') || 43);
  const initialFrame = Math.min(800, Math.max(0, Number(params.get('frame') || 0)));
  const lastFrame = () => meta ? meta.steps.length - 1 : 800;
  const frameAtStep = step => Math.max(0, meta.steps.findIndex(value => value >= step));
  const fmt = x => Number(x).toFixed(5);
  const pct = x => `${(100 * x).toFixed(x < .001 ? 5 : 1)}%`;
  function probability(x) { return x < 1e-6 ? `${(x * 100).toExponential(2)}%` : pct(x); }
  function element(tag, text, className) { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (className) node.className = className; return node; }
  function stop() { if (timer) clearInterval(timer); timer = null; $('play').textContent = 'Play ▶'; }
  function shareLink() {
    const p = new URLSearchParams({scenario: $('environment').value, seed: $('repeat').value, frame: $('step').value});
    history.replaceState(null, '', `/selector-explorer?${p}`);
    $('frame-download').href = `/api/selector-explorer?${p}`;
  }
  function syncRule() {
    const scenario = $('environment').value;
    $('rule-heading').textContent = rules[scenario][0]; $('rule-description').textContent = rules[scenario][1];
    $('world-warning').textContent = scenario.startsWith('practice')
      ? 'Each selector has its own evolving world: its past choices decide which means have fallen. The response is a declared simulation assumption, calibrated to the observed #771 curve.'
      : scenario === 'fixed' ? 'All three selectors face the same fixed means and matched batch draws. This is an empirical reward-table simulation.'
      : 'All three selectors face the same changing means and matched batch draws. The change is a declared simulation rule, not a new measurement of the LM.';
    $('before').hidden = $('after').hidden = !['gradual', 'abrupt'].includes(scenario);
  }
  function buildCards() {
    for (const method of methods) {
      const card = element('article', undefined, 'method-card'); card.dataset.method = method;
      const header = element('header'); const h = element('h3', names[method]); h.style.color = methodColors[method]; header.append(h);
      header.append(element('p', method === 'predictor'
        ? 'Predict the gap from settings. Three steps exploit; every fourth explores the complete pool.'
        : method === 'reinforce' ? 'Sample from learned probabilities. REINFORCE updates them from the observed gap; entropy = 0.'
        : 'Choose independently with probability 0.1% for every tokenizer. No reward learning.', 'method-note'));
      card.append(header);
      for (const kind of ['world', 'belief']) {
        const wrap = element('div'); const heading = element('div', undefined, 'scatter-heading');
        heading.append(element('strong', kind === 'world' ? 'TRUE HARDNESS' : method === 'predictor' ? 'PREDICTED HARDNESS' : 'SAMPLING PROBABILITY'));
        heading.append(element('small', kind === 'world' ? 'Hidden from selector' : 'Before this choice'));
        wrap.append(heading);
        if (kind === 'belief' && method === 'reinforce') {
          const label = element('label', undefined, 'log-control'); const input = element('input'); input.type = 'checkbox'; input.id = 'log-probabilities';
          input.addEventListener('change', drawAll); label.append(input, document.createTextNode('Log scale for tiny probabilities')); wrap.append(label);
        }
        const canvas = element('canvas', undefined, `${kind}-scatter`); canvas.id = `${method}-${kind}`;
        canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', `${names[method]} ${kind === 'world' ? 'true mean rewards' : 'pre-choice estimates'} for all 1,000 tokenizers`);
        wrap.append(canvas);
        if (kind === 'belief') { const note = element('p', '', 'belief-status'); note.id = `${method}-belief-status`; wrap.append(note); }
        card.append(wrap);
      }
      const choice = element('div', undefined, 'choice-block'); choice.id = `${method}-choice`; card.append(choice);
      $('method-cards').append(card);
    }
    for (const method of methods) for (const kind of ['world', 'belief']) {
      const canvas = $(`${method}-${kind}`);
      canvas.addEventListener('mousemove', event => hoverPoint(canvas, method, kind, event));
      canvas.addEventListener('mouseleave', () => { $('point-tooltip').hidden = true; });
      canvas.addEventListener('click', event => { const index = closestPoint(canvas, event); if (index !== null) inspect(index); });
    }
    for (const method of methods) {
      const label = element('span'); const swatch = element('i', undefined, 'dot'); swatch.style.background = methodColors[method]; label.append(swatch, document.createTextNode(names[method])); $('method-legend').append(label);
    }
  }
  function context(canvas) {
    const box = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
    const width = Math.max(100, box.width), height = Math.max(80, box.height);
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext('2d'); ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.font = '9px Manrope, sans-serif'; return {ctx, width, height};
  }
  function scatter(method, kind) {
    const canvas = $(`${method}-${kind}`), {ctx, width, height} = context(canvas);
    const row = current.methods[method], values = kind === 'world' ? row.means : row.beliefs;
    const left = 39, right = width - 9, top = 13, bottom = height - 28;
    const logarithmic = kind === 'belief' && method === 'reinforce' && $('log-probabilities').checked;
    canvas.points = null;
    if (!values) { ctx.fillStyle = '#737373'; ctx.fillText('Reconstructing exact selector estimates…', 9, height / 2); return; }
    const isProbability = kind === 'belief' && method !== 'predictor';
    const transform = value => logarithmic ? Math.log10(Math.max(1e-30, value)) : value;
    let low = 0, high = isProbability ? 1 : .10;
    if (logarithmic) { low = -30; high = 0; }
    else if (kind === 'belief' && method === 'predictor') { low = Math.min(0, Math.floor(Math.min(...values) * 100) / 100); high = Math.max(.10, Math.ceil(Math.max(...values) * 100) / 100); }
    const y = value => bottom - (transform(value) - low) / (high - low) * (bottom - top);
    ctx.strokeStyle = '#E9EDF1'; ctx.fillStyle = '#737373'; ctx.textAlign = 'right';
    for (let tick = 0; tick <= 4; tick++) {
      const v = low + (high - low) * tick / 4, yy = bottom - tick / 4 * (bottom - top);
      ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(right, yy); ctx.stroke();
      const label = logarithmic ? `10^${Math.round(v)}` : isProbability ? `${Math.round(v * 100)}%` : v.toFixed(3);
      ctx.fillText(label, left - 5, yy + 3);
    }
    const points = [];
    meta.order.forEach((index, rank) => {
      const xx = left + (rank + .5) / 1000 * (right - left), yy = y(values[index]);
      points.push({x: xx, y: yy, index});
      ctx.fillStyle = colors[meta.tokenizers[index].algorithm]; ctx.globalAlpha = .48;
      ctx.beginPath(); ctx.arc(xx, yy, 1.65, 0, Math.PI * 2); ctx.fill();
    }); ctx.globalAlpha = 1;
    const byIndex = new Map(points.map(p => [p.index, p]));
    if (kind === 'world') {
      const p = byIndex.get(row.best); ctx.fillStyle = '#B38A29'; ctx.beginPath(); ctx.moveTo(p.x, p.y - 6); ctx.lineTo(p.x + 6, p.y); ctx.lineTo(p.x, p.y + 6); ctx.lineTo(p.x - 6, p.y); ctx.closePath(); ctx.fill();
    }
    for (const [index, radius, color] of [[selected, 7, '#2d68a2'], [row.selected, 5, '#242424']]) {
      const p = byIndex.get(index); ctx.strokeStyle = color; ctx.lineWidth = 1.7; ctx.beginPath(); ctx.arc(p.x, p.y, radius, 0, Math.PI * 2); ctx.stroke();
    }
    let count = 0; ctx.textAlign = 'center'; ctx.fillStyle = '#737373';
    for (const name of ['bpe', 'wordpiece', 'unigram']) {
      const n = meta.tokenizers.filter(t => t.algorithm === name).length;
      ctx.fillText(name === 'bpe' ? 'BPE' : name === 'wordpiece' ? 'WordPiece' : 'Unigram', left + (count + n / 2) / 1000 * (right - left), height - 10); count += n;
    }
    canvas.points = points;
  }
  function closestPoint(canvas, event) {
    if (!canvas.points) return null;
    const rect = canvas.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
    let best = null, distance = 13 * 13;
    for (const p of canvas.points) { const d = (p.x - x) ** 2 + (p.y - y) ** 2; if (d < distance) { distance = d; best = p.index; } }
    return best;
  }
  function hoverPoint(canvas, method, kind, event) {
    const index = closestPoint(canvas, event), tooltip = $('point-tooltip');
    if (index === null) { tooltip.hidden = true; return; }
    const t = meta.tokenizers[index], row = current.methods[method];
    tooltip.textContent = `#${index} · ${t.algorithm} · ${(t.vocab_size / 1000).toFixed(0)}k\nTrue mean: ${fmt(row.means[index])} BPB\n${method === 'predictor' ? 'Predicted gap: ' : 'Probability: '}${row.beliefs ? method === 'predictor' ? `${fmt(row.beliefs[index])} BPB` : probability(row.beliefs[index]) : 'reconstructing…'}\nPreviously selected ${row.counts[index]} times\nClick to follow this tokenizer`;
    tooltip.hidden = false; tooltip.style.left = `${Math.min(event.clientX + 13, innerWidth - 290)}px`; tooltip.style.top = `${Math.max(8, Math.min(event.clientY + 13, innerHeight - 155))}px`;
  }
  function metric(label, value) { const item = element('div'); item.append(element('dt', label), element('dd', value)); return item; }
  function renderChoice(method) {
    const row = current.methods[method], box = $(`${method}-choice`), t = meta.tokenizers[row.selected]; box.replaceChildren();
    box.append(element('p', `Chose #${row.selected} · ${t.algorithm} · rank ${row.rank}/1,000`, 'choice-title'), element('p', row.mode, 'choice-mode'));
    const metrics = element('dl', undefined, 'step-metrics');
    metrics.append(metric('CHOSEN TRUE MEAN', fmt(row.chosen_mean)), metric('REWARD IT OBSERVES', fmt(row.observed_reward)), metric('HARDEST TRUE MEAN', fmt(row.best_mean)), metric('TOKENIZERS VISITED BEFORE', row.visited_before_step.toLocaleString())); box.append(metrics);
    box.append(element('p', `${fmt(row.best_mean)} − ${fmt(row.chosen_mean)} = ${fmt(row.missed_hardness)} BPB missed hardness`, 'shortfall-equation'));
    box.append(element('p', `The observed reward includes ${row.noise >= 0 ? '+' : ''}${fmt(row.noise)} BPB of measured batch noise. The selector receives this one observation, not all the dots above.`, 'recent-history'));
    const after = method === 'predictor' ? 'One squared-error update changes the shared gap predictor.' : method === 'reinforce' ? 'One REINFORCE update changes the shared sampling network.' : 'No selector weights are updated.';
    box.append(element('p', `${after} ${row.post_step_changes ? `Then the simulation makes ${row.post_step_changes === 1 ? 'this tokenizer' : `all ${row.post_step_changes} tokenizers in this family`} a little easier.` : 'This choice itself does not change any true means in this environment.'}`, 'after-choice'));
    box.append(element('p', `Last ${row.recent_steps.toLocaleString()} choices, including this one: largest one-tokenizer share ${pct(row.largest_recent_share)}. Most selected: ${row.recent_top.map(x => `#${x.index} (${x.count})`).join(', ')}.`, 'recent-history'));
    $(`${method}-belief-status`).textContent = row.belief_error || (method === 'predictor' ? 'These are estimated BPB gaps, not probabilities. The black ring shows the actual saved choice.' : 'Each dot is a selection probability. The black ring shows the sampled choice.');
  }
  function drawTimeline() {
    const canvas = $('timeline'), {ctx, width, height} = context(canvas);
    const left = 42, right = width - 14, top = 13, bottom = height - 24;
    const max = Math.max(.005, ...methods.flatMap(method => current.methods[method].curve)) * 1.12;
    const xx = step => left + step / 20000 * (right - left), yy = value => bottom - value / max * (bottom - top);
    ctx.textAlign = 'right'; ctx.fillStyle = '#737373'; ctx.strokeStyle = '#E9EDF1';
    for (let i = 0; i <= 4; i++) { const v = i / 4 * max, y = yy(v); ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke(); ctx.fillText(v.toFixed(3), left - 6, y + 3); }
    ctx.textAlign = 'center'; for (const step of [0, 5000, 10000, 15000, 20000]) ctx.fillText(step.toLocaleString(), xx(step), height - 7);
    if (['gradual', 'abrupt'].includes(current.scenario)) { ctx.strokeStyle = '#B9A8C9'; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(xx(10000), top); ctx.lineTo(xx(10000), bottom); ctx.stroke(); ctx.setLineDash([]); }
    for (const method of methods) { ctx.strokeStyle = methodColors[method]; ctx.lineWidth = 1.8; ctx.beginPath(); current.methods[method].curve.forEach((v, i) => { const x = xx((i + .5) * 100), y = yy(v); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); ctx.stroke(); }
    ctx.strokeStyle = '#242424'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(xx(current.step), top); ctx.lineTo(xx(current.step), bottom); ctx.stroke();
    canvas.range = {left, right};
  }
  function renderInspector() {
    if (!current) return; const t = meta.tokenizers[selected]; $('tokenizer-id').value = selected;
    $('tokenizer-settings').textContent = `#${selected} · ${t.algorithm} · ${t.vocab_size.toLocaleString()} vocabulary · ${t.pretokenizer} · ${t.noise_per_million === null ? 'merge noise does not apply' : `${(100 * t.noise_per_million / 1000000).toFixed(1)}% merge noise`} · starting mean ${fmt(t.initial_mean)} BPB`;
    const total = Object.values(t.domain_bytes).reduce((a, b) => a + b, 0); $('tokenizer-domains').replaceChildren(...Object.entries(t.domain_bytes).map(([key, value]) => element('span', `${key}: ${(100 * value / total).toFixed(1)}%`)));
    $('tokenizer-comparison').replaceChildren(...methods.map(method => { const row = current.methods[method], tr = element('tr');
      const belief = row.beliefs ? method === 'predictor' ? `${fmt(row.beliefs[selected])} BPB` : probability(row.beliefs[selected]) : 'Reconstructing…';
      for (const text of [names[method], `${fmt(row.means[selected])} BPB`, belief, row.counts[selected].toLocaleString(), row.selected === selected ? 'Yes' : '—']) tr.append(element('td', text)); return tr; }));
  }
  function inspect(index) { selected = Math.min(999, Math.max(0, Math.round(index))); renderInspector(); drawAll(); }
  function renderTable() {
    if (!$('all-tokenizers').open || !current) return;
    const method = $('table-method').value, sort = $('table-sort').value, row = current.methods[method];
    const signature = `${current.scenario}-${current.seed}-${current.step}-${method}-${sort}`; if (signature === tableSignature) return; tableSignature = signature;
    const order = Array.from({length: 1000}, (_, i) => i).sort((a, b) => sort === 'index' ? a - b : sort === 'visits' ? row.counts[b] - row.counts[a] || a - b : sort === 'belief' && row.beliefs ? row.beliefs[b] - row.beliefs[a] || a - b : row.means[b] - row.means[a] || a - b);
    $('pool-rows').replaceChildren(...order.map(index => { const t = meta.tokenizers[index], tr = element('tr'); tr.tabIndex = 0;
      const belief = row.beliefs ? method === 'predictor' ? `${fmt(row.beliefs[index])} BPB` : probability(row.beliefs[index]) : 'Reconstructing…';
      for (const text of [`#${index}`, t.algorithm, t.vocab_size.toLocaleString(), fmt(t.initial_mean), fmt(row.means[index]), belief, row.counts[index].toLocaleString()]) tr.append(element('td', text));
      const choose = () => { inspect(index); $('inspect-heading').scrollIntoView({behavior: 'smooth', block: 'center'}); }; tr.addEventListener('click', choose); tr.addEventListener('keydown', event => { if (event.key === 'Enter') choose(); }); return tr; }));
  }
  function drawAll() { if (!current) return; drawTimeline(); for (const method of methods) for (const kind of ['world', 'belief']) scatter(method, kind); }
  async function loadFrame() {
    if (!meta) return; const chosenFrame = Number($('step').value); const generation = ++version;
    if (request) request.abort(); request = new AbortController(); syncRule(); shareLink();
    $('playback-state').textContent = 'Reading saved observations…';
    const p = new URLSearchParams({scenario: $('environment').value, seed: $('repeat').value, frame: String(chosenFrame)});
    try {
      const response = await fetch(`/api/selector-explorer?${p}`, {signal: request.signal}); if (!response.ok) throw new Error(`Saved data request returned ${response.status}`);
      const data = await response.json(); if (generation !== version) return; current = data;
      $('step-label').textContent = `Entering step ${data.step.toLocaleString()} / 20,000`;
      $('step').setAttribute('aria-valuetext', `Step ${data.step.toLocaleString()}`);
      $('playback-state').textContent = methods.every(method => data.methods[method].beliefs) ? 'Exact decisions · estimates before this reward' : 'Decisions loaded · exact estimates still preparing';
      $('explorer-error').hidden = true;
      for (const method of methods) renderChoice(method); renderInspector(); renderTable(); drawAll();
    } catch (error) { if (error.name === 'AbortError' || generation !== version) return; stop(); $('explorer-error').hidden = false; $('explorer-error').textContent = `Could not load this saved step. ${error.message}`; }
  }
  function changeFrame(frame) { $('step').value = Math.min(lastFrame(), Math.max(0, frame)); loadFrame(); }
  $('step').addEventListener('input', () => { stop(); if (pendingFrame) cancelAnimationFrame(pendingFrame); pendingFrame = requestAnimationFrame(loadFrame); });
  for (const id of ['environment', 'repeat']) $(id).addEventListener('change', () => { stop(); loadFrame(); });
  $('start').addEventListener('click', () => { stop(); changeFrame(0); }); $('before').addEventListener('click', () => { stop(); changeFrame(frameAtStep(9900)); });
  $('after').addEventListener('click', () => { stop(); changeFrame(frameAtStep(11500)); }); $('end').addEventListener('click', () => { stop(); changeFrame(lastFrame()); });
  $('play').addEventListener('click', () => { if (timer) { stop(); return; } if (Number($('step').value) === lastFrame()) changeFrame(0);
    $('play').textContent = 'Pause Ⅱ'; timer = setInterval(() => { const next = Math.min(lastFrame(), Number($('step').value) + Number($('speed').value)); changeFrame(next); if (next === lastFrame()) stop(); }, 500); });
  $('timeline').addEventListener('click', event => { if (!current) return; stop(); const box = $('timeline').getBoundingClientRect(), range = $('timeline').range; changeFrame(Math.round((event.clientX - box.left - range.left) / (range.right - range.left) * lastFrame())); });
  $('tokenizer-id').addEventListener('change', () => { const number = Number($('tokenizer-id').value); if (Number.isFinite(number)) inspect(number); });
  $('all-tokenizers').addEventListener('toggle', renderTable); for (const id of ['table-method', 'table-sort']) $(id).addEventListener('change', renderTable);
  let resizeTimer; window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(drawAll, 80); });
  async function initialize() {
    try {
      const response = await fetch('/api/selector-explorer?view=meta'); if (!response.ok) throw new Error('Simulation metadata unavailable'); meta = await response.json();
      for (const scenario of meta.scenarios) { const option = element('option', scenario.name); option.value = scenario.id; $('environment').append(option); }
      for (const seed of meta.seeds) { const option = element('option', `Repeat ${seed - 42} · seed ${seed}`); option.value = String(seed); $('repeat').append(option); }
      $('environment').value = rules[initialScenario] ? initialScenario : 'practice_family'; $('repeat').value = meta.seeds.includes(initialSeed) ? String(initialSeed) : '43'; $('step').max = lastFrame(); $('step').value = Math.min(lastFrame(),initialFrame);
      buildCards(); $('step').disabled = $('play').disabled = false;
      $('calibration-note').textContent = `Changing-world rules use #771’s observed gap drop from ${fmt(meta.calibration.start_gap)} to ${fmt(meta.calibration.end_gap)} BPB over 1,500 training updates and 1,381 selections. That observational curve does not identify how any other tokenizer would really respond.`;
      await document.fonts.ready; await loadFrame();
    } catch (error) { $('explorer-error').hidden = false; $('explorer-error').textContent = error.message; }
  }
  initialize();
})();
