/* Raised front end. No framework, no build step. Keyboard-first; every view has a non-visual equivalent. */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const state = { result: null, fig: null, sample: null };

const STEPS = [
  ['pages', 'Reading the document', 'pdfium'],
  ['find', 'Finding the charts', 'Nemotron 3.5 Super VL'],
  ['semantics', 'Reading each chart three times', 'Nemotron 3.5 Super VL'],
  ['measure', 'Measuring the pixels', 'code'],
  ['adjudicate', 'Settling disagreements by redrawing', 'code'],
  ['verify', 'Comparing the redraw with the original', 'Nemotron 3.5 Super VL'],
  ['describe', 'Writing descriptions, auditing captions', 'Nemotron 3 Ultra'],
  ['guard', 'Checking every number and sentence', 'code + Nemotron 3.5 Lightning'],
  ['tactile', 'Tactile graphic and braille', 'liblouis'],
];

function toast(msg, ms = 4000) {
  const t = $('#toast'); t.textContent = msg; t.hidden = false;
  clearTimeout(toast.h); toast.h = setTimeout(() => (t.hidden = true), ms);
}
function fmt(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return 'not read';
  if (typeof v !== 'number') return String(v);
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toLocaleString('en-GB', { maximumFractionDigits: 2 }) + ' million';
  if (a >= 1000) return Math.round(v).toLocaleString('en-GB');
  return Number(v.toPrecision(digits)).toLocaleString('en-GB');
}

/* ---------------- start: samples and upload ---------------- */
async function loadSamples() {
  const list = await fetch('/api/samples').then(r => r.json()).catch(() => []);
  $('#samples').innerHTML = list.map(s => `<div role="listitem"><button class="sample" type="button" data-sample="${esc(s.id)}">
      <img src="${esc(s.thumb)}" alt="">
      <span><b>${esc(s.title)}</b><small>${esc(s.subtitle)}</small></span></button></div>`).join('');
}
$('#samples').addEventListener('click', e => {
  const b = e.target.closest('[data-sample]'); if (!b) return;
  run({ sample: b.dataset.sample });
});
const fileInput = $('#file');
fileInput.addEventListener('change', () => { $('#go').disabled = !fileInput.files.length; });
$('#upload').addEventListener('submit', e => { e.preventDefault(); if (fileInput.files[0]) run({ file: fileInput.files[0] }); });
const drop = $('#upload');
['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add('dragging'); }));
['dragleave', 'drop'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove('dragging'); }));
drop.addEventListener('drop', e => {
  const f = e.dataTransfer.files[0]; if (!f) return;
  const dt = new DataTransfer(); dt.items.add(f); fileInput.files = dt.files; $('#go').disabled = false;
});

/* ---------------- run with live progress (server-sent events over fetch) ---------------- */
async function run({ sample, file }) {
  const fd = new FormData();
  if (sample) fd.append('sample', sample); else fd.append('file', file, file.name);
  $('#results').hidden = true; $('#figure').hidden = true;
  startRail();
  const ctrl = new AbortController(); const tmo = setTimeout(() => ctrl.abort(), 300000);
  try {
    const r = await fetch('/api/run-stream', { method: 'POST', body: fd, signal: ctrl.signal });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
    const reader = r.body.getReader(), dec = new TextDecoder(); let buf = '', result = null;
    while (!result) {
      const { value, done } = await reader.read();
      if (done) throw new Error('the connection closed early');
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 2);
        if (!line.startsWith('data: ')) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.error) throw new Error(ev.error);
        if (ev.result) { result = ev.result; break; }
        railEvent(ev);
      }
    }
    await railDone();
    showResults(result);
  } catch (e) {
    await railDone(false);
    toast(e.name === 'AbortError' ? 'That took too long. Try a sample, or a shorter document.' : 'Could not read the document: ' + e.message, 7000);
  } finally { clearTimeout(tmo); }
}

let railQ = Promise.resolve();
function startRail() {
  $('#progress').hidden = false;
  $('#rail').innerHTML = STEPS.map(([k, t, who]) => `<li data-step="${k}"><span class="dot" aria-hidden="true"></span><span>${t} <span class="info"></span></span><span class="who">${who}</span></li>`).join('');
  railQ = Promise.resolve();
  $('#progress').scrollIntoView({ block: 'start' });
}
function railEvent(ev) {
  railQ = railQ.then(() => new Promise(done => {
    const li = $(`#rail li[data-step="${ev.step}"]`); if (!li) return done();
    li.classList.toggle('running', ev.status === 'running');
    if (ev.status === 'done') li.classList.add('done');
    const bits = [];
    if (ev.of) bits.push(`${ev.n ?? 0} of ${ev.of}`);
    if (ev.figures !== undefined) bits.push(`${ev.figures} chart${ev.figures === 1 ? '' : 's'}`);
    if (ev.pages !== undefined) bits.push(`${ev.pages} page${ev.pages === 1 ? '' : 's'}`);
    if (ev.usd !== undefined) bits.push(`$${Number(ev.usd).toFixed(4)} so far`);
    if (ev.recorded) bits.push('recorded');
    if (bits.length) $('.info', li).textContent = '· ' + bits.join(' · ');
    if (ev.status === 'done') $('#progress-live').textContent = `${$('span', li).firstChild.textContent.trim()}: done.`;
    setTimeout(done, ev.recorded ? 90 : 0);  // recorded runs arrive at once; pace them so they can be followed
  }));
}
async function railDone(ok = true) {
  await railQ;
  $$('#rail li').forEach(li => li.classList.remove('running'));
  if (ok) setTimeout(() => { $('#progress').hidden = true; }, 400);
}

/* ---------------- results grid ---------------- */
function showResults(res) {
  state.result = res;
  const b = $('#banner');
  b.hidden = !res.recorded;
  if (res.recorded) b.textContent = `Recorded result: ${res.reason || 'this sample was read earlier with the same pipeline'}. Nothing was sent to the models for this view.`;
  const n = res.figures.length, v = res.figures.filter(f => f.status === 'verified').length, r = res.figures.filter(f => f.status === 'review').length;
  $('#summary').textContent = `${n} chart${n === 1 ? '' : 's'}: ${v} verified, ${r} to review${res.cost ? ` · cost $${Number(res.cost.total_usd || 0).toFixed(4)}` : ''}`;
  const ex = $('#export'); ex.hidden = !res.export; if (res.export) ex.href = res.export;
  $('#figures').innerHTML = res.figures.map((f, i) => `<li><button class="fig-card" type="button" data-fig="${i}">
      <img src="${esc(f.image)}" alt="">
      <span class="body">
        <span class="badge ${esc(f.status)}">${{ verified: 'Verified', review: 'Needs review', unsupported: 'Not a supported chart' }[f.status] || esc(f.status)}</span>
        <span class="title">${esc(f.figure_number ? 'Figure ' + f.figure_number + ': ' : '')}${esc(f.title || f.caption || 'Untitled chart')}</span>
        ${f.reason ? `<span class="reason">${esc(f.reason)}</span>` : ''}
      </span></button></li>`).join('');
  $('#results').hidden = false;
  $('#results').scrollIntoView({ block: 'start' });
  $('#results-h').focus?.();
}
$('#figures').addEventListener('click', e => { const b = e.target.closest('[data-fig]'); if (b) openFigure(+b.dataset.fig); });
$('#fig-close').addEventListener('click', () => { $('#figure').hidden = true; $(`[data-fig="${state.result.figures.indexOf(state.fig)}"]`)?.focus(); });

/* ---------------- figure view ---------------- */
function openFigure(i) {
  const f = state.fig = state.result.figures[i];
  $('#fig-kicker').textContent = [f.page ? `Page ${f.page}` : '', f.chart_type ? f.chart_type.replace('_', ' ') + ' chart' : ''].filter(Boolean).join(' · ');
  $('#fig-h').textContent = (f.figure_number ? `Figure ${f.figure_number}: ` : '') + (f.title || f.caption || 'Untitled chart');
  $('#fig-status').innerHTML = `<span class="badge ${esc(f.status)}">${{ verified: 'Verified', review: 'Needs review', unsupported: 'Not a supported chart' }[f.status]}</span> ${esc(f.reason || '')}`;
  renderCompare(f); renderTable(f); renderExplorer(f); renderDescribe(f); renderTactile(f);
  selectTab('tab-compare');
  const sec = $('#figure'); sec.hidden = false; sec.scrollIntoView({ block: 'start' }); sec.focus();
}

/* tabs: WAI-ARIA tabs pattern with arrow-key navigation */
const tabs = $$('#tabs [role=tab]');
function selectTab(id) {
  tabs.forEach(t => {
    const on = t.id === id;
    t.setAttribute('aria-selected', on); t.tabIndex = on ? 0 : -1;
    $('#' + t.getAttribute('aria-controls')).hidden = !on;
  });
}
tabs.forEach((t, i) => {
  t.addEventListener('click', () => selectTab(t.id));
  t.addEventListener('keydown', e => {
    const k = { ArrowRight: 1, ArrowLeft: -1, Home: -i, End: tabs.length - 1 - i }[e.key];
    if (k === undefined) return;
    e.preventDefault(); const n = tabs[(i + k + tabs.length) % tabs.length]; n.focus(); selectTab(n.id);
  });
});

/* Check tab */
function renderCompare(f) {
  $('#img-original').src = f.image; $('#img-original').alt = f.description?.alt_text || 'Original chart';
  $('#img-rerender').src = f.rerender || ''; $('#img-rerender').hidden = !f.rerender;
  $('#iou').textContent = f.iou !== undefined && f.iou !== null ? `Redraw overlaps the original's marks: ${(100 * f.iou).toFixed(0)}% (intersection over union).` : '';
  const disputed = (f.points || []).filter(p => p.status === 'disputed');
  $('#no-disputes').hidden = disputed.length > 0 || f.status === 'unsupported';
  $('#dispute-list').innerHTML = disputed.map((p, k) => `<li class="dispute">
      <p><b>${esc(p.series)}, ${esc(p.x)}</b>: the vision model read <b>${fmt(p.reading_a)}</b>, the pixels measure <b>${fmt(p.reading_b)}</b>. ${esc(p.note || '')}</p>
      <label>Value to use <input type="text" inputmode="decimal" data-k="${k}" value="${p.value ?? ''}" aria-label="Value for ${esc(p.series)} at ${esc(p.x)}"></label></li>`).join('');
  $('#accept').hidden = disputed.length === 0;
}
$('#mix').addEventListener('input', e => { $('#img-rerender').style.opacity = e.target.value / 100; $('#mix-out').textContent = e.target.value + '%'; });
$('#accept').addEventListener('click', async () => {
  const f = state.fig, disputed = (f.points || []).filter(p => p.status === 'disputed');
  const values = $$('#dispute-list input').map(inp => {
    const p = disputed[+inp.dataset.k]; return { series: p.series, x: p.x, value: parseFloat(inp.value.replace(/,/g, '')) };
  });
  if (values.some(v => Number.isNaN(v.value))) { toast('Every value must be a number.'); return; }
  $('#accept').disabled = true;
  try {
    const r = await fetch(`/api/figure/${encodeURIComponent(state.result.job)}/${encodeURIComponent(f.id)}/accept`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values }) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
    const nf = await r.json();
    const i = state.result.figures.indexOf(f); state.result.figures[i] = nf;
    showResults(state.result); openFigure(i); toast('Accepted. Table, description, sound and tactile graphic updated.');
  } catch (e) { toast('Could not accept: ' + e.message); }
  $('#accept').disabled = false;
});

/* Table tab */
function renderTable(f) {
  const t = f.table;
  if (!t || !t.x?.length) { $('#table-wrap').innerHTML = '<p class="fine" style="padding:.8rem">No data table for this figure.</p>'; $('#csv').hidden = true; return; }
  const names = Object.keys(t.series);
  const flagged = new Set((f.points || []).filter(p => p.status === 'disputed').map(p => p.series + '|' + p.x));
  const xTitle = f.semantics?.x_title || (f.chart_type?.includes('bar') ? 'Category' : 'x');
  $('#table-wrap').innerHTML = `<table><caption>${esc(f.title || 'Data')}${f.semantics?.y_title ? ' (' + esc(f.semantics.y_title) + ')' : ''}</caption>
    <thead><tr><th scope="col">${esc(xTitle)}</th>${names.map(n => `<th scope="col">${esc(n)}</th>`).join('')}</tr></thead>
    <tbody>${t.x.map((x, i) => `<tr><th scope="row">${esc(typeof x === 'number' ? Number(x.toFixed(4)) : x)}</th>${names.map(n => {
      const v = t.series[n][i]; const fl = flagged.has(n + '|' + x);
      return `<td class="${fl ? 'flag' : ''}">${v === null || v === undefined ? '<span aria-label="not read">–</span>' : esc(fmt(v, 4))}${fl ? '<span class="sr-only"> (needs review)</span>' : ''}</td>`;
    }).join('')}</tr>`).join('')}</tbody></table>`;
  $('#csv').hidden = !f.csv; if (f.csv) $('#csv').href = f.csv;
}

/* Explore tab: arrow keys step through points; pitch follows the value; the same mapping as the downloadable WAV */
let audio = null, ex = { s: 0, i: 0 };
function mappingFor(f) {
  if (f.sound?.series) return f.sound;
  const t = f.table, all = Object.values(t.series).flat().filter(v => v !== null && v !== undefined);
  const lo = Math.min(...all), hi = Math.max(...all), log = lo > 0 && hi / lo > 50;
  const hz = v => { if (v === null || v === undefined) return null; const r = log ? (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo) || 1) : (v - lo) / (hi - lo || 1); return 220 * Math.pow(4, r); };
  return { low_hz: 220, high_hz: 880, seconds_per_point: 0.25, series: Object.fromEntries(Object.entries(t.series).map(([n, vs]) => [n, vs.map((v, i) => ({ x: t.x[i], value: v, hz: hz(v) }))])) };
}
function renderExplorer(f) {
  const ok = f.table && f.table.x?.length;
  $('#tab-explore').hidden = !ok; if (!ok) return;
  ex = { s: 0, i: 0, map: mappingFor(f), names: Object.keys(f.table.series) };
  drawExplorer(); $('#wav').hidden = !f.wav; if (f.wav) $('#wav').href = f.wav;
  $('#explorer-now').textContent = `${ex.names.length} series, ${f.table.x.length} points. Press an arrow key to start.`;
}
function drawExplorer() {
  const f = state.fig, W = 800, H = 260, P = 24, t = f.table;
  const vals = Object.values(t.series).flat().filter(v => v !== null && v !== undefined);
  const lo = Math.min(...vals), hi = Math.max(...vals), n = t.x.length;
  const X = i => P + (W - 2 * P) * (n === 1 ? 0.5 : i / (n - 1)), Y = v => H - P - (H - 2 * P) * ((v - lo) / (hi - lo || 1));
  const colours = ['var(--accent)', 'var(--ok)', 'var(--warn)', 'var(--bad)'];
  let svg = '';
  ex.names.forEach((name, s) => {
    const pts = t.series[name].map((v, i) => v === null || v === undefined ? null : [X(i), Y(v)]).filter(Boolean);
    svg += `<polyline fill="none" stroke="${colours[s % 4]}" stroke-width="${s === ex.s ? 4 : 2}" opacity="${s === ex.s ? 1 : 0.35}" points="${pts.map(p => p.join(',')).join(' ')}"/>`;
  });
  const v = t.series[ex.names[ex.s]][ex.i];
  if (v !== null && v !== undefined) svg += `<circle cx="${X(ex.i)}" cy="${Y(v)}" r="9" fill="var(--focus)" stroke="var(--ink)" stroke-width="2"/>`;
  $('#explorer-svg').innerHTML = svg;
}
function tone(hz, ms = 180) {
  if (!$('#sound-on').checked || !hz) return;
  audio = audio || new (window.AudioContext || window.webkitAudioContext)();
  const o = audio.createOscillator(), g = audio.createGain(), t0 = audio.currentTime;
  o.type = 'triangle'; o.frequency.value = hz;
  g.gain.setValueAtTime(0, t0); g.gain.linearRampToValueAtTime(0.25, t0 + 0.01); g.gain.exponentialRampToValueAtTime(0.0001, t0 + ms / 1000);
  o.connect(g).connect(audio.destination); o.start(t0); o.stop(t0 + ms / 1000 + 0.02);
}
function announce() {
  const f = state.fig, name = ex.names[ex.s], pt = ex.map.series[name][ex.i];
  const unit = f.semantics?.y_title ? ` ${f.semantics.y_title}` : '';
  const msg = `${pt.x}: ${fmt(pt.value)}${pt.value === null ? '' : unit}. ${name}, point ${ex.i + 1} of ${f.table.x.length}.`;
  $('#explorer-now').textContent = msg; $('#explorer-live').textContent = msg; tone(pt.hz); drawExplorer();
}
let playing = null;
$('#explorer').addEventListener('keydown', e => {
  const n = state.fig?.table?.x.length; if (!n) return;
  const k = e.key;
  if (k === 'ArrowRight' || k === 'ArrowUp') ex.i = Math.min(n - 1, ex.i + 1);
  else if (k === 'ArrowLeft' || k === 'ArrowDown') ex.i = Math.max(0, ex.i - 1);
  else if (k === 'Home') ex.i = 0;
  else if (k === 'End') ex.i = n - 1;
  else if (k === 's' || k === 'S') { ex.s = (ex.s + 1) % ex.names.length; $('#explorer-live').textContent = `Series: ${ex.names[ex.s]}.`; drawExplorer(); e.preventDefault(); return; }
  else if (k === ' ') {
    e.preventDefault(); clearInterval(playing); let i = 0;
    const pts = ex.map.series[ex.names[ex.s]], step = (ex.map.seconds_per_point || 0.25) * 1000;
    $('#explorer-live').textContent = `Playing ${ex.names[ex.s]}.`;
    playing = setInterval(() => { if (i >= pts.length) return clearInterval(playing); ex.i = i; tone(pts[i].hz, step * 0.9); drawExplorer(); i++; }, step);
    return;
  } else return;
  e.preventDefault(); announce();
});

/* Description tab */
function renderDescribe(f) {
  const d = f.description || {};
  $('#desc-long').innerHTML = (d.long_description || 'No description.').split(/\n{2,}/).map(p => `<p>${esc(p)}</p>`).join('');
  $('#desc-guide').innerHTML = (d.exploration_guide || []).map(s => `<li>${esc(s)}</li>`).join('');
  $('#desc-audit').innerHTML = (d.audit || []).length ? d.audit.map(a => `<li class="${esc(a.verdict)}"><b>${esc(a.verdict)}</b>: “${esc(a.claim_quote)}”${a.explanation ? `. ${esc(a.explanation)}` : ''}</li>`).join('') : '<li>No claims about this figure were found in the text.</li>';
  $('#desc-alt').textContent = d.alt_text || '';
}

/* Tactile tab */
function renderTactile(f) {
  const t = f.tactile || {};
  $('#tactile-preview').innerHTML = t.preview_svg || '<p class="fine" style="padding:1rem">No tactile graphic for this figure.</p>';
  $('#tactile-key').innerHTML = (t.key || []).length ? '<b>Key:</b> ' + t.key.map(k => esc(k)).join(' · ') : '';
  for (const [id, url] of [['#tactile-pdf', t.pdf], ['#tactile-svg', t.svg], ['#brf', f.brf]]) { $(id).hidden = !url; if (url) $(id).href = url; }
}

/* ---------------- boot ---------------- */
(async function boot() {
  loadSamples();
  try {
    const h = await fetch('/api/health').then(r => r.json());
    const c = $('#mode'); c.textContent = h.live ? 'Live on Nebius Token Factory' : 'Recorded mode'; c.classList.toggle('live', !!h.live);
  } catch { $('#mode').textContent = 'offline'; }
})();
