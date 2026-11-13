'use strict';
const $ = (s) => document.querySelector(s);
const C = { text: '#e7e8ea', text2: '#a3a7ae', muted: '#6b7079', line: '#1f2125', line2: '#2a2d32', panel: '#111215',
  accent: '#7aa7ff', base: '#c9a26b', green: '#5cc59a', red: '#f0716b', solo: '#8b919a', flat: '#5fb8b0', amber: '#e3b341' };

// ---------------- tabs ----------------
function showTab() {
  const id = (location.hash || '#playground').slice(1);
  const name = document.getElementById('tab-' + id) ? id : 'playground';
  document.querySelectorAll('section.tab').forEach(s => s.classList.toggle('on', s.id === 'tab-' + name));
  document.querySelectorAll('nav a').forEach(a => a.classList.toggle('on', a.dataset.tab === name));
  if (name === 'results') drawResults();
  window.scrollTo(0, 0);
}
window.addEventListener('hashchange', showTab);

// ---------------- formula lab ----------------
const PRESETS = [
  ['Hand-built peer factors', [
    ['Sector momentum', 'NbrMean($ret20, G_sector)'],
    ['Within-sector 5-day reversal', 'NbrRel($ret5, G_sector)'],
    ['12-1 month momentum (no peers)', '$mom12_1'],
  ]],
  ['Found by PeerAlpha (with graph operators)', [
    ['Sector rank of 20-day return, plus close vs VWAP', 'Add(Greater(Mul($cvwap, 0.01), -1), CsRank(NbrRank($ret20, G_sector)))'],
    ['Peer rank of 5-day return, scaled by 60-day return', 'Mul(NbrRank($ret5, G_sector), Add(0.5, $ret60))'],
    ['Sector 120-day return minus a momentum-VWAP rank', 'Sub(NbrMean($ret120, G_sector), CsRank(Mul(WMA($cvwap, 10), $mom12_1)))'],
    ['Smoothed peer rank of 5-day return', 'Log(Greater(EMA(NbrRank(Sub($ret5, 5), G_sector), 40), $ret))'],
    ['Momentum gated by short-term rank', 'Mul($mom12_1, Less(CsRank($ret20), Mul(-0.5, EMA($cvwap, 10))))'],
  ]],
  ['Found without graph operators', [
    ['Volatility vs range-weighted momentum', 'Sub(0.01, Sub($vol20, Mul($range, Sub($mom12_1, CsRank($ret)))))'],
    ['Up-day share vs long momentum', 'Greater(Mean(Greater($ret, $vol20), 20), Abs(Add($mom12_1, $ret60)))'],
  ]],
];
const GROUPS = [
  ['Features', t => t.kind === 'feat'],
  ['Peers', t => t.kind === 'graph' || t.kind === 'gop'],
  ['Time-series', t => t.kind === 'rolling' || t.kind === 'pair'],
  ['Element-wise', t => t.kind === 'unary' || t.kind === 'binary'],
  ['Windows (days)', t => t.kind === 'win'],
  ['Constants', t => t.kind === 'const'],
];
const DESC = {
  ret: '1-day return', ret5: '5-day return', ret20: '20-day return', ret60: '60-day return', ret120: '120-day return',
  mom12_1: '12-1 month momentum: return from t-250 to t-20', cvwap: 'close / VWAP - 1', range: 'high / low - 1', vol20: '20-day volatility',
  sector: 'sector graph: up to 8 same-industry peers',
  NbrMean: 'mean of x over the stock\'s peers', NbrRel: 'x minus the peer mean', NbrRank: 'share of peers with lower x',
  Ref: 'x, w days ago', Mean: 'rolling mean', Sum: 'rolling sum', Std: 'rolling std', Max: 'rolling max', Min: 'rolling min',
  Delta: 'x minus x w days ago', WMA: 'linearly weighted mean', EMA: 'exponential mean', Cov: 'rolling covariance of x, y', Corr: 'rolling correlation of x, y',
  Abs: '|x|', Log: 'log |x|', CsRank: 'rank across all stocks that day, 0 to 1',
  Add: 'x + y', Sub: 'x - y', Mul: 'x * y', Div: 'x / y', Greater: 'max(x, y)', Less: 'min(x, y)',
};
const label = (t) => t.kind === 'feat' ? '$' + t.name : t.kind === 'graph' ? 'G_' + t.name : t.name;

let mk = null, builder = new PA.Builder(), current = null, pinned = null;

function buildPalette() {
  const pal = $('#palette');
  pal.innerHTML = GROUPS.map(([g, f], gi) => `<div style="margin-bottom:12px"><div class="lbl">${g}</div><div class="toks">${
    PA.TOKENS.filter(f).map(t => `<button class="tok k-${t.kind}" data-id="${t.id}" title="${DESC[t.name] || t.kind}">${label(t)}</button>`).join('')
  }</div></div>`).join('');
  pal.addEventListener('click', (e) => {
    const b = e.target.closest('.tok');
    if (!b || b.disabled) return;
    builder.push(PA.TOKENS[+b.dataset.id]);
    $('#preset').value = '';
    update();
  });
  const sel = $('#preset');
  sel.innerHTML = '<option value="">Choose a formula…</option>' + PRESETS.map(([g, items]) =>
    `<optgroup label="${g}">${items.map(([n, f]) => `<option value="${f}">${n}</option>`).join('')}</optgroup>`).join('');
  sel.addEventListener('change', () => { if (sel.value) load(sel.value); });
}

function load(src) {
  const toks = PA.parse(src);
  builder = PA.Builder.from(toks, Math.max(PA.MAX_LEN, toks.length));
  update();
}

const NEED = {
  '': 'Start with a feature, or a constant.',
};
function stackText() {
  const types = builder.types();
  if (!types.length) return 'Empty. Start with a feature (or a constant).';
  if (builder.complete()) return '<b>Finished.</b> One complete expression on the stack. Keep going to wrap it in another operator.';
  const top = types[types.length - 1];
  const hint = {
    win: 'a window is on top, so the next token must be a time-series operator',
    graph: 'a graph is on top, so the next token must be a peer operator',
    const: 'a constant needs a feature next, then a binary operator',
    expr: 'combine the open pieces with an operator',
  }[top];
  return `Stack: <b>${types.join(' · ')}</b>. ${hint[0].toUpperCase() + hint.slice(1)}.`;
}

function update() {
  const mask = builder.mask();
  document.querySelectorAll('.tok').forEach(b => { b.disabled = !mask[+b.dataset.id]; });
  const n = builder.tokens.length;
  $('#undo').disabled = !n; $('#clear').disabled = !n;
  $('#post').innerHTML = builder.tokens.map(t => `<span class="chip">${label(t)}</span>`).join('') +
    `<span class="n">${n} / ${builder.limit} tokens</span>`;
  $('#stackline').innerHTML = stackText();
  const node = builder.node();
  if (node) $('#infix').textContent = PA.show(node);
  else if (n) $('#infix').innerHTML = builder.stack.map(s => PA.show(s.node)).join('<span class="ph">  ·  </span>') + ' <span class="ph">(unfinished)</span>';
  else $('#infix').innerHTML = '<span class="ph">Pick tokens on the left, or load a formula.</span>';
  scoreNow(node);
}

let scoreTimer = null;
function scoreNow(node) {
  clearTimeout(scoreTimer);
  if (!mk) return;
  if (!node) { current = null; render(); return; }
  $('#status').textContent = 'Scoring…';
  scoreTimer = setTimeout(() => {
    const t0 = performance.now();
    const s = PA.score(node, mk);
    current = s && { ...s, name: PA.show(node), roll: rollMean(s.daily, 120) };
    $('#status').textContent = `Scored on ${mk.N} stocks × ${mk.T.toLocaleString()} days in ${Math.round(performance.now() - t0)} ms, in your browser.`;
    render();
  }, 20);
}

function rollMean(x, w) {
  const o = new Float64Array(x.length).fill(NaN);
  for (let t = w - 1; t < x.length; t++) {
    let s = 0, n = 0;
    for (let k = t - w + 1; k <= t; k++) if (Number.isFinite(x[k])) { s += x[k]; n++; }
    if (n >= w * 0.6) o[t] = s / n;
  }
  return o;
}

const fmt = (v, d = 3) => Number.isFinite(v) ? (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d) : '–';
const cls = (v) => !Number.isFinite(v) ? '' : v > 0 ? 'pos' : 'neg';
const PERIODS = ['2018–19', '2020–21', '2022–24'];

function render() {
  const s = current;
  const cell = (k, v, sub, big) => `<div class="${big ? 'big' : ''}"><div class="k">${k}</div><div class="v ${cls(v)}">${fmt(v)}</div><div class="s">${sub}</div></div>`;
  if (!s) {
    $('#scores').innerHTML = cell('mean test IC', NaN, builder.tokens.length ? 'finish the formula to score it' : 'no formula yet', true) +
      PERIODS.map(p => cell(p, NaN, '')).join('');
  } else {
    $('#scores').innerHTML = cell('mean test IC', s.mean, `3 test periods · ${Math.round(s.coverage * 100)}% of cells defined`, true) +
      s.windows.map((w, i) => cell(PERIODS[i], w.ic, `t ${Number.isFinite(w.t) ? w.t.toFixed(1) : '–'}`)).join('');
  }
  $('#pin').disabled = !s; $('#unpin').disabled = !pinned;
  drawIC();
  $('#narr').innerHTML = narrate(s);
}

function narrate(s) {
  if (!s) return 'The agent sees exactly this palette: at every step, tokens that could not lead to a valid formula within 10 tokens are masked.';
  if (!Number.isFinite(s.mean)) return 'This formula is undefined on most days (for example a constant, or a division by zero), so it has no IC.';
  const signs = s.windows.map(w => Math.sign(w.ic));
  const steady = signs.every(x => x === signs[0]);
  const a = Math.abs(s.mean);
  const size = a >= 0.03 ? 'a strong' : a >= 0.015 ? 'a useful' : a >= 0.007 ? 'a weak' : 'almost no';
  let txt = `<b>${size[0].toUpperCase() + size.slice(1)} signal</b>`;
  txt += steady ? `, with the same sign in all three test periods.` : `, but its sign changes between test periods, which the pool's sign gate would reject.`;
  if (s.mean < 0 && steady) txt += ' Negative IC is fine: the pool would hold it with a negative weight.';
  if (/G_sector/.test(s.name)) txt += ' It reads the stock\'s sector peers.';
  return txt;
}

function drawIC() {
  const svg = $('#icchart');
  const W = 900, H = 240, L = 44, R = 10, T = 12, B = 24;
  if (!mk) { svg.innerHTML = ''; return; }
  const [lo] = mk.windows[0], hi = mk.windows[mk.windows.length - 1][1];
  const series = [];
  if (pinned) series.push([pinned, C.base, 1.5]);
  if (current) series.push([current, C.accent, 2]);
  let ymax = 0.08;
  for (const [s] of series) for (let t = lo; t < hi; t++) if (Number.isFinite(s.roll[t])) ymax = Math.max(ymax, Math.abs(s.roll[t]) * 1.1);
  ymax = Math.ceil(ymax * 50) / 50;
  const sx = (t) => L + (t - lo) / (hi - 1 - lo) * (W - L - R);
  const sy = (v) => T + (1 - (v + ymax) / (2 * ymax)) * (H - T - B);
  let g = '';
  mk.windows.forEach(([a, b], i) => {
    g += `<rect x="${sx(a)}" y="${T}" width="${sx(b - 1) - sx(a)}" height="${H - T - B}" fill="${i % 2 ? '#131417' : '#0f1012'}"/>`;
    g += `<text x="${(sx(a) + sx(b - 1)) / 2}" y="${H - 6}" text-anchor="middle">${PERIODS[i]}</text>`;
  });
  const step = ymax > 0.1 ? 0.05 : 0.02;
  for (let v = -Math.floor(ymax / step) * step; v <= ymax + 1e-9; v += step) {
    const z = Math.abs(v) < 1e-9;
    g += `<line x1="${L}" x2="${W - R}" y1="${sy(v)}" y2="${sy(v)}" stroke="${z ? C.line2 : C.line}" ${z ? '' : 'stroke-dasharray="2 4"'}/>`;
    g += `<text x="${L - 8}" y="${sy(v) + 3}" text-anchor="end">${(z ? 0 : v).toFixed(2)}</text>`;
  }
  for (const [s, col, w] of series) {
    let d = '', pen = false;
    for (let t = lo; t < hi; t += 2) {
      const v = s.roll[t];
      if (!Number.isFinite(v)) { pen = false; continue; }
      d += `${pen ? 'L' : 'M'}${sx(t).toFixed(1)},${sy(Math.max(-ymax, Math.min(ymax, v))).toFixed(1)}`;
      pen = true;
    }
    g += `<path d="${d}" fill="none" stroke="${col}" stroke-width="${w}" stroke-linejoin="round"/>`;
    s.windows.forEach((wd, i) => {
      const [a, b] = mk.windows[i];
      if (Number.isFinite(wd.ic)) g += `<line x1="${sx(a)}" x2="${sx(b - 1)}" y1="${sy(wd.ic)}" y2="${sy(wd.ic)}" stroke="${col}" stroke-width="1" stroke-dasharray="5 4" opacity=".7"><title>${PERIODS[i]} mean IC ${fmt(wd.ic)}</title></line>`;
    });
  }
  if (!series.length) g += `<text x="${(L + W) / 2}" y="${H / 2}" text-anchor="middle" style="font-size:12px">finish a formula to see its IC over time</text>`;
  svg.innerHTML = g;
  const short = (n) => n.length > 48 ? n.slice(0, 46) + '…' : n;
  $('#legend').innerHTML = series.map(([s, col]) => `<span title="${s.name}"><i style="background:${col}"></i>${short(s.name)}</span>`).reverse().join('') +
    (series.length ? `<span><i style="background:${C.text2};height:0;border-top:1px dashed ${C.text2}"></i>period mean</span>` : '');
}

function randomFormula() {
  for (let attempt = 0; attempt < 50; attempt++) {
    const b = new PA.Builder();
    while (true) {
      const mask = b.mask(), ok = PA.TOKENS.filter((t, i) => mask[i]);
      if (b.complete() && (b.tokens.length >= 3 && Math.random() < 0.4 || !ok.length)) break;
      if (!ok.length) break;
      b.push(ok[Math.floor(Math.random() * ok.length)]);
    }
    if (b.complete()) return b;
  }
  return new PA.Builder();
}

function buildPeers() {
  const m = mk.meta, sel = $('#stock');
  const order = m.codes.map((c, i) => i).sort((a, b) => m.codes[a].localeCompare(m.codes[b]));
  sel.innerHTML = order.map(i => `<option value="${i}">${m.codes[i]} · ${m.sectors[m.sector_ids[i]]}</option>`).join('');
  const start = m.codes.indexOf('HDFCBANK');
  sel.value = String(start >= 0 ? start : order[0]);
  const draw = () => {
    const i = +sel.value, nb = m.sector_nbrs[i];
    $('#peers').innerHTML = nb.length ? nb.map((j, r) => `<div class="peer"><span class="i">${r + 1}</span><span class="c">${m.codes[j]}</span><span class="s">${m.sectors[m.sector_ids[j]]}</span></div>`).join('')
      : '<div class="peer"><span></span><span class="s" style="text-align:left">No peers: this stock is alone in its industry here, so peer operators give no value for it.</span><span></span></div>';
  };
  sel.addEventListener('change', draw);
  draw();
}

async function init() {
  buildPalette();
  $('#undo').onclick = () => { builder = PA.Builder.from(builder.tokens.slice(0, -1), builder.limit); $('#preset').value = ''; update(); };
  $('#clear').onclick = () => { builder = new PA.Builder(); $('#preset').value = ''; update(); };
  $('#random').onclick = () => { builder = randomFormula(); $('#preset').value = ''; update(); };
  $('#pin').onclick = () => { pinned = current; render(); };
  $('#unpin').onclick = () => { pinned = null; render(); };
  update();
  try {
    const [meta, buf] = await Promise.all([
      fetch('assets/market.json').then(r => r.json()),
      fetch('assets/market.bin').then(r => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); }),
    ]);
    mk = new PA.Market(meta, buf);
    $('#loader').remove();
    buildPeers();
    // start with sector momentum pinned and the within-sector reversal shown
    load('NbrMean($ret20, G_sector)');
    pinned = PA.score(builder.node(), mk);
    pinned = { ...pinned, name: 'NbrMean($ret20, G_sector)', roll: rollMean(pinned.daily, 120) };
    $('#preset').value = 'NbrRel($ret5, G_sector)';
    load('NbrRel($ret5, G_sector)');
  } catch (e) {
    $('#status').textContent = 'Could not load the market data (' + e.message + '). The formula builder still works.';
    $('#loader').remove();
  }
}

// ---------------- results ----------------
const N200 = [
  // name, [w0, w1, w2], mean, ls sharpe, color, kind
  ['Hand factor + PeerAlpha pool (blend)', [0.070, 0.027, 0.053], 0.050, null, C.accent],
  ['Hand factor + random-search pool (blend)', [0.067, 0.029, 0.046], 0.047, null, C.flat],
  ['Hand factor: sector momentum, reversal, 12-1 momentum', [0.063, 0.025, 0.053], 0.047, 1.19, C.base],
  ['PeerAlpha, sector graph, hand-factor prior', [0.068, 0.025, 0.042], 0.045, 1.50, '#4f7bd6'],
  ['Random search, same vocabulary and prior', [0.063, 0.030, 0.040], 0.044, 1.40, C.solo],
  ['PeerAlpha, sector graph, no prior', [0.063, 0.027, 0.032], 0.041, 1.46, null],
  ['Hand factor: sector momentum and reversal only', [0.045, 0.028, 0.043], 0.038, 0.93, null],
  ['PeerAlpha without graph operators', [0.058, 0.007, 0.029], 0.031, 0.55, C.red],
];
const N100 = [
  ['Hand factor: sector momentum, reversal, 12-1 momentum', [0.055, 0.034, 0.043], 0.044],
  ['Hand factor + PeerAlpha pool (blend)', [0.053, 0.039, 0.039], 0.044],
  ['PeerAlpha, sector graph, hand-factor prior', [0.047, 0.038, 0.034], 0.040],
  ['Hand factor: sector momentum and reversal only', [0.047, 0.044, 0.027], 0.039],
  ['Random search, same vocabulary and prior', [0.045, 0.036, 0.034], 0.038],
  ['PeerAlpha without graph operators', [0.041, 0.005, 0.028], 0.025],
];

function table(rows, ls) {
  const best = Math.max(...rows.map(r => r[2]));
  const bestLS = ls ? Math.max(...rows.map(r => r[3] ?? -1)) : 0;
  return `<tr><th>Method</th>${PERIODS.map(p => `<th>${p}</th>`).join('')}<th>Mean</th>${ls ? '<th>LS Sharpe</th>' : ''}</tr>` +
    rows.map(r => `<tr${r[2] === best ? ' class="hl"' : ''}><td>${r[0]}</td>${r[1].map(v => `<td>${v.toFixed(3)}</td>`).join('')}` +
      `<td class="${r[2] === best ? 'best' : ''}">${r[2].toFixed(3)}</td>` +
      (ls ? `<td class="${r[3] === bestLS ? 'best' : ''}">${r[3] == null ? 'n/a' : r[3].toFixed(2)}</td>` : '') + '</tr>').join('');
}

function drawResults() {
  if ($('#rchart').childElementCount) return;
  $('#t200').innerHTML = table(N200, true);
  $('#t100').innerHTML = table(N100, false);
  const rows = N200.filter(r => r[4]);
  const W = 900, H = 260, L = 44, R = 10, T = 10, B = 26, ymax = 0.08;
  const groups = [...PERIODS, 'Mean'];
  const gw = (W - L - R) / groups.length, bw = Math.min(22, (gw - 24) / rows.length);
  const sy = (v) => T + (1 - v / ymax) * (H - T - B);
  let g = '';
  for (let v = 0; v <= ymax + 1e-9; v += 0.02) g += `<line x1="${L}" x2="${W - R}" y1="${sy(v)}" y2="${sy(v)}" stroke="${v ? C.line : C.line2}"/><text x="${L - 8}" y="${sy(v) + 3}" text-anchor="end">${v.toFixed(2)}</text>`;
  groups.forEach((name, gi) => {
    const x0 = L + gi * gw + (gw - bw * rows.length) / 2;
    if (gi === 3) g += `<line x1="${L + gi * gw}" x2="${L + gi * gw}" y1="${T}" y2="${H - B}" stroke="${C.line2}"/>`;
    rows.forEach((r, i) => {
      const v = gi < 3 ? r[1][gi] : r[2];
      g += `<rect x="${x0 + i * bw + 1}" y="${sy(v)}" width="${bw - 2}" height="${sy(0) - sy(v)}" rx="2" fill="${r[4]}"><title>${r[0]} · ${name}: ${v.toFixed(3)}</title></rect>`;
    });
    g += `<text x="${L + gi * gw + gw / 2}" y="${H - 8}" text-anchor="middle" style="${gi === 3 ? 'fill:' + C.text2 : ''}">${name}</text>`;
  });
  $('#rchart').innerHTML = g;
  $('#rkeys').innerHTML = rows.map(r => `<span><i style="background:${r[4]};height:8px;width:8px;border-radius:2px"></i>${r[0]}</span>`).join('');
}

showTab();
init();
