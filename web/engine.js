// PeerAlpha formula engine for the browser: vocabulary, action mask, evaluation and IC.
// A line-by-line port of peeralpha/{vocab,grammar,ops,expr,metrics}.py for the v5 configuration
// (9 features, 7 windows, 23 operators, sector graph, formulas of at most 10 tokens).
(function (root) {
  'use strict';
  const NAN = NaN, EPS = 1e-8;

  // ---------------- vocabulary (configs/v5_sector_prior.yaml) ----------------
  const FEATURES = ['ret', 'ret5', 'ret20', 'ret60', 'ret120', 'mom12_1', 'cvwap', 'range', 'vol20'];
  const CONSTANTS = [-5, -2, -1, -0.5, -0.01, 0.01, 0.5, 1, 2, 5];
  const WINDOWS = [1, 5, 10, 20, 40, 60, 120];
  const GRAPHS = ['sector'];
  const OPS = [
    ['Abs', 'unary'], ['Log', 'unary'], ['CsRank', 'unary'],
    ['Add', 'binary'], ['Sub', 'binary'], ['Mul', 'binary'], ['Div', 'binary'], ['Greater', 'binary'], ['Less', 'binary'],
    ['Ref', 'rolling'], ['Mean', 'rolling'], ['Sum', 'rolling'], ['Std', 'rolling'], ['Max', 'rolling'], ['Min', 'rolling'],
    ['Delta', 'rolling'], ['WMA', 'rolling'], ['EMA', 'rolling'],
    ['Cov', 'pair'], ['Corr', 'pair'],
    ['NbrMean', 'gop'], ['NbrRel', 'gop'], ['NbrRank', 'gop'],
  ];
  const MAX_LEN = 10;

  const TOKENS = [];
  const add = (kind, name, value = 0) => TOKENS.push({ id: TOKENS.length, kind, name, value });
  for (const [n, k] of OPS) add(k, n);
  for (const f of FEATURES) add('feat', f);
  for (const c of CONSTANTS) add('const', String(c), c);
  for (const w of WINDOWS) add('win', String(w), w);
  for (const g of GRAPHS) add('graph', g);
  const BY_KEY = {};
  for (const t of TOKENS) BY_KEY[keyOf(t)] = t;
  function keyOf(t) { return ['feat', 'const', 'win', 'graph'].includes(t.kind) ? `${t.kind}:${t.name}` : t.name; }
  const ARITY = { unary: 1, binary: 2, rolling: 2, gop: 2, pair: 3 };

  // ---------------- grammar: postfix stack + action mask (grammar.py) ----------------
  class Builder {
    constructor(limit = MAX_LEN) { this.limit = limit; this.tokens = []; this.stack = []; }
    types() { return this.stack.map(s => s.type); }
    complete() { return this.stack.length === 1 && this.stack[0].type === 'expr'; }
    node() { return this.complete() ? this.stack[0].node : null; }
    valid(tok) {
      const types = this.types(), m = types.length, top = m ? types[m - 1] : null;
      const used = this.tokens.length, hasExpr = types.includes('expr');
      const fits = (mAfter, exprAfter) => used + 1 + (mAfter - 1) + (exprAfter ? 0 : 2) <= this.limit;
      switch (tok.kind) {
        case 'feat': return (top === null || top === 'expr' || top === 'const') && fits(m + 1, true);
        case 'const': return (top === null || top === 'expr') && fits(m + 1, hasExpr);
        case 'win': return top === 'expr' && fits(m + 1, true);
        case 'graph': return top === 'expr' && fits(m + 1, true);
        case 'unary': return top === 'expr' && fits(m, true);
        case 'binary': return m >= 2 && ['expr', 'const'].includes(types[m - 1]) && ['expr', 'const'].includes(types[m - 2])
          && !(types[m - 1] === 'const' && types[m - 2] === 'const') && fits(m - 1, true);
        case 'rolling': return m >= 2 && types[m - 1] === 'win' && types[m - 2] === 'expr' && fits(m - 1, true);
        case 'pair': return m >= 3 && types[m - 1] === 'win' && types[m - 2] === 'expr' && types[m - 3] === 'expr' && fits(m - 2, true);
        case 'gop': return m >= 2 && types[m - 1] === 'graph' && types[m - 2] === 'expr' && fits(m - 1, true);
      }
      return false;
    }
    mask() { return TOKENS.map(t => this.valid(t)); }
    push(tok) {
      const k = tok.kind;
      if (k === 'feat') this.stack.push({ type: 'expr', node: { kind: 'feat', name: tok.name } });
      else if (k === 'const') this.stack.push({ type: 'const', node: { kind: 'const', value: tok.value } });
      else if (k === 'win') this.stack.push({ type: 'win', node: { kind: 'win', value: tok.value } });
      else if (k === 'graph') this.stack.push({ type: 'graph', node: { kind: 'graph', name: tok.name } });
      else {
        const a = ARITY[k];
        if (this.stack.length < a) throw new Error(`${tok.name} needs ${a} operands`);
        const kids = this.stack.splice(this.stack.length - a, a);
        this.stack.push({ type: 'expr', node: { kind: 'op', name: tok.name, op: k, args: kids.map(c => c.node) } });
      }
      this.tokens.push(tok);
    }
    static from(tokens, limit) { const b = new Builder(limit); for (const t of tokens) b.push(t); return b; }
  }

  function show(n) {
    if (!n) return '';
    if (n.kind === 'feat') return '$' + n.name;
    if (n.kind === 'const' || n.kind === 'win') return String(n.value);
    if (n.kind === 'graph') return 'G_' + n.name;
    return `${n.name}(${n.args.map(show).join(', ')})`;
  }

  // Parse "NbrMean($ret20, G_sector)" into postfix tokens.
  function parse(src) {
    let i = 0;
    const s = src.replace(/\s+/g, '');
    const out = [];
    function expr() {
      let m = /^\$([A-Za-z0-9_]+)/.exec(s.slice(i));
      if (m) { i += m[0].length; out.push(BY_KEY['feat:' + m[1]]); return; }
      m = /^G_([a-z]+)/.exec(s.slice(i));
      if (m) { i += m[0].length; out.push(BY_KEY['graph:' + m[1]]); return; }
      m = /^-?[0-9.]+/.exec(s.slice(i));
      if (m) { i += m[0].length; return 'num:' + m[0]; }
      m = /^([A-Za-z]+)\(/.exec(s.slice(i));
      if (!m) throw new Error('cannot parse at ' + s.slice(i));
      i += m[0].length;
      const op = BY_KEY[m[1]];
      if (!op) throw new Error('unknown operator ' + m[1]);
      const nargs = ARITY[op.kind];
      for (let a = 0; a < nargs; a++) {
        if (a) { if (s[i] !== ',') throw new Error('expected ,'); i++; }
        const r = expr();
        if (r) {          // a number: a window when it is the last argument of a rolling / pair op
          const v = r.slice(4);
          const isWin = (op.kind === 'rolling' || op.kind === 'pair') && a === nargs - 1;
          const t = BY_KEY[(isWin ? 'win:' : 'const:') + String(Number(v))];
          if (!t) throw new Error(`${v} is not in the vocabulary`);
          out.push(t);
        }
      }
      if (s[i] !== ')') throw new Error('expected )');
      i++;
      out.push(op);
    }
    const r = expr();
    if (r) out.push(BY_KEY['const:' + String(Number(r.slice(4)))]);
    if (i !== s.length) throw new Error('trailing input');
    return out;
  }

  // ---------------- market ----------------
  class Market {
    constructor(meta, buf) {
      const T = meta.T, N = meta.N, TN = T * N;
      this.meta = meta; this.T = T; this.N = N;
      const close = new Float64Array(new Float32Array(buf, 0, TN));
      const cvwap = new Float64Array(new Float32Array(buf, TN * 4, TN));
      const range = new Float64Array(new Float32Array(buf, TN * 8, TN));
      const back = (a, b = 0) => {
        const o = new Float64Array(TN).fill(NAN);
        for (let t = a; t < T; t++) for (let i = 0; i < N; i++) o[t * N + i] = close[(t - b) * N + i] / close[(t - a) * N + i] - 1;
        return o;
      };
      const ret = back(1);
      this.feat = { ret, ret5: back(5), ret20: back(20), ret60: back(60), ret120: back(120), mom12_1: back(250, 20),
        cvwap, range, vol20: rollStd(ret, 20, T, N) };
      const H = meta.horizon, y = new Float64Array(TN).fill(NAN);
      for (let t = 0; t + H < T; t++) for (let i = 0; i < N; i++) y[t * N + i] = close[(t + H) * N + i] / close[t * N + i] - 1;
      this.target = y;
      this.nbrs = meta.sector_nbrs;
      this.windows = meta.tests.map(([a, b]) => [meta.dates.findIndex(d => d >= a), lastIdx(meta.dates, b)]);
    }
    graph() { return this.nbrs; }
  }
  function lastIdx(dates, b) { let k = -1; for (let j = 0; j < dates.length; j++) if (dates[j] <= b) k = j; return k + 1; }

  // ---------------- operators (ops.py); arrays are [T*N] row-major, NaN propagates ----------------
  function rolling(x, w, T, N, f) {
    const o = new Float64Array(T * N).fill(NAN), buf = new Float64Array(w);
    for (let i = 0; i < N; i++) for (let t = w - 1; t < T; t++) {
      for (let k = 0; k < w; k++) buf[k] = x[(t - w + 1 + k) * N + i];
      o[t * N + i] = f(buf);
    }
    return o;
  }
  function rollStd(x, w, T, N) {
    if (w < 2) return new Float64Array(T * N).fill(NAN);
    return rolling(x, w, T, N, b => { let m = 0; for (const v of b) m += v; m /= w; let s = 0; for (const v of b) s += (v - m) ** 2; return Math.sqrt(s / w); });
  }
  const weighted = (wts) => (b) => { let s = 0; for (let k = 0; k < b.length; k++) s += b[k] * wts[k]; return s; };
  function normed(a) { const s = a.reduce((p, q) => p + q, 0); return a.map(v => v / s); }

  const ROLL = {
    Ref: (x, w, T, N) => { const o = new Float64Array(T * N).fill(NAN); if (w <= 0) return x.slice(); for (let j = w * N; j < T * N; j++) o[j] = x[j - w * N]; return o; },
    Mean: (x, w, T, N) => rolling(x, w, T, N, b => { let s = 0; for (const v of b) s += v; return s / w; }),
    Sum: (x, w, T, N) => rolling(x, w, T, N, b => { let s = 0; for (const v of b) s += v; return s; }),
    Std: rollStd,
    Max: (x, w, T, N) => rolling(x, w, T, N, b => { let m = -Infinity; for (const v of b) { if (v !== v) return NAN; if (v > m) m = v; } return m; }),
    Min: (x, w, T, N) => rolling(x, w, T, N, b => { let m = Infinity; for (const v of b) { if (v !== v) return NAN; if (v < m) m = v; } return m; }),
    Delta: (x, w, T, N) => { const r = ROLL.Ref(x, w, T, N); for (let j = 0; j < r.length; j++) r[j] = x[j] - r[j]; return r; },
    WMA: (x, w, T, N) => rolling(x, w, T, N, weighted(normed(Array.from({ length: w }, (_, k) => k + 1)))),
    EMA: (x, w, T, N) => { const a = 2 / (w + 1); return rolling(x, w, T, N, weighted(normed(Array.from({ length: w }, (_, k) => (1 - a) ** (w - 1 - k))))); },
  };
  function pairRoll(x, y, w, T, N, corr) {
    const o = new Float64Array(T * N).fill(NAN);
    if (w < 2) return o;
    for (let i = 0; i < N; i++) for (let t = w - 1; t < T; t++) {
      let mx = 0, my = 0;
      for (let k = t - w + 1; k <= t; k++) { mx += x[k * N + i]; my += y[k * N + i]; }
      mx /= w; my /= w;
      let c = 0, vx = 0, vy = 0;
      for (let k = t - w + 1; k <= t; k++) { const dx = x[k * N + i] - mx, dy = y[k * N + i] - my; c += dx * dy; vx += dx * dx; vy += dy * dy; }
      c /= w;
      if (!corr) { o[t * N + i] = c; continue; }
      const den = Math.sqrt(vx / w) * Math.sqrt(vy / w);
      o[t * N + i] = den < EPS ? NAN : c / den;
    }
    return o;
  }
  const BIN = {
    Add: (a, b) => a + b, Sub: (a, b) => a - b, Mul: (a, b) => a * b,
    Div: (a, b) => (Math.abs(b) < EPS ? NAN : a / b),
    Greater: (a, b) => (a !== a || b !== b ? NAN : Math.max(a, b)),
    Less: (a, b) => (a !== a || b !== b ? NAN : Math.min(a, b)),
  };
  function rankRows(x, T, N) {
    const o = new Float64Array(T * N).fill(NAN), idx = [];
    for (let t = 0; t < T; t++) {
      idx.length = 0;
      for (let i = 0; i < N; i++) if (Number.isFinite(x[t * N + i])) idx.push(i);
      idx.sort((p, q) => x[t * N + p] - x[t * N + q] || p - q);
      const n = Math.max(idx.length, 2);
      idx.forEach((i, r) => { o[t * N + i] = r / (n - 1); });
    }
    return o;
  }
  function zscoreRows(a, T, N, minValid) {
    const o = new Float64Array(T * N).fill(NAN);
    for (let t = 0; t < T; t++) {
      let n = 0, s = 0;
      for (let i = 0; i < N; i++) { const v = a[t * N + i]; if (Number.isFinite(v)) { n++; s += v; } }
      if (n < minValid) continue;
      const m = s / n; let q = 0;
      for (let i = 0; i < N; i++) { const v = a[t * N + i]; if (Number.isFinite(v)) q += (v - m) ** 2; }
      const sd = Math.sqrt(q / n);
      if (sd < 1e-10) continue;
      for (let i = 0; i < N; i++) { const v = a[t * N + i]; if (Number.isFinite(v)) o[t * N + i] = (v - m) / Math.max(sd, 1e-12); }
    }
    return o;
  }
  function graphOp(name, x, nb, T, N) {
    const o = new Float64Array(T * N).fill(NAN);
    for (let t = 0; t < T; t++) for (let i = 0; i < N; i++) {
      const own = x[t * N + i];
      let cnt = 0, s = 0, below = 0;
      for (const j of nb[i]) { const v = x[t * N + j]; if (Number.isFinite(v)) { cnt++; s += v; if (v < own) below++; } }
      if (name === 'NbrMean') o[t * N + i] = cnt > 0 ? s / cnt : NAN;
      else if (name === 'NbrRel') o[t * N + i] = own - (cnt > 0 ? s / cnt : NAN);
      else o[t * N + i] = cnt > 0 && Number.isFinite(own) ? below / cnt : NAN;
    }
    return o;
  }

  function evaluate(node, mk) {
    const T = mk.T, N = mk.N;
    if (node.kind === 'feat') return mk.feat[node.name];
    if (node.kind === 'const') return node.value;
    const a = node.args;
    switch (node.op) {
      case 'unary': {
        const x = evaluate(a[0], mk);
        if (node.name === 'CsRank') return rankRows(x, T, N);
        const o = new Float64Array(x.length);
        if (node.name === 'Abs') for (let j = 0; j < x.length; j++) o[j] = Math.abs(x[j]);
        else for (let j = 0; j < x.length; j++) o[j] = x[j] !== x[j] ? NAN : Math.log(Math.max(Math.abs(x[j]), 1e-6));
        return o;
      }
      case 'binary': {
        const x = evaluate(a[0], mk), y = evaluate(a[1], mk), f = BIN[node.name];
        const L = typeof x === 'number' ? y.length : x.length, o = new Float64Array(L);
        const xs = typeof x === 'number', ys = typeof y === 'number';
        for (let j = 0; j < L; j++) o[j] = f(xs ? x : x[j], ys ? y : y[j]);
        return o;
      }
      case 'rolling': return ROLL[node.name](evaluate(a[0], mk), a[1].value, T, N);
      case 'pair': return pairRoll(evaluate(a[0], mk), evaluate(a[1], mk), a[2].value, T, N, node.name === 'Corr');
      case 'gop': return graphOp(node.name, zscoreRows(evaluate(a[0], mk), T, N, 2), mk.graph(a[1].name), T, N);
    }
    throw new Error('cannot evaluate ' + node.kind);
  }

  // ---------------- metrics (metrics.py) ----------------
  function dailyIC(a, y, T, N, minObs = 20) {
    const ic = new Float64Array(T).fill(NAN);
    for (let t = 0; t < T; t++) {
      let n = 0, sa = 0, sy = 0;
      for (let i = 0; i < N; i++) { const p = a[t * N + i], q = y[t * N + i]; if (Number.isFinite(p) && Number.isFinite(q)) { n++; sa += p; sy += q; } }
      if (n < minObs) continue;
      const ma = sa / n, my = sy / n; let c = 0, va = 0, vy = 0;
      for (let i = 0; i < N; i++) {
        const p = a[t * N + i], q = y[t * N + i];
        if (Number.isFinite(p) && Number.isFinite(q)) { c += (p - ma) * (q - my); va += (p - ma) ** 2; vy += (q - my) ** 2; }
      }
      if (va < 1e-10 || vy < 1e-10) continue;
      ic[t] = c / Math.max(Math.sqrt(va * vy), 1e-12);
    }
    return ic;
  }
  function meanFinite(x, lo = 0, hi = x.length) {
    let s = 0, n = 0;
    for (let j = lo; j < hi; j++) if (Number.isFinite(x[j])) { s += x[j]; n++; }
    return n ? s / n : NAN;
  }
  function neweyWestT(x, lo, hi, lags) {
    const v = []; for (let j = lo; j < hi; j++) if (Number.isFinite(x[j])) v.push(x[j]);
    const n = v.length; if (n < 3) return NAN;
    const m = v.reduce((p, q) => p + q, 0) / n, d = v.map(q => q - m);
    let varr = d.reduce((p, q) => p + q * q, 0) / n;
    for (let l = 1; l <= Math.min(lags, n - 1); l++) {
      let s = 0; for (let j = l; j < n; j++) s += d[j] * d[j - l];
      varr += 2 * (1 - l / (lags + 1)) * (s / (n - l)) * (n - l) / n;
    }
    return m / Math.sqrt(Math.max(varr, 1e-18) / n);
  }

  // Score a formula on the three walk-forward test windows.
  function score(node, mk) {
    const sig = evaluate(node, mk);
    if (typeof sig === 'number') return null;
    const ic = dailyIC(sig, mk.target, mk.T, mk.N);
    const wins = mk.windows.map(([lo, hi]) => ({ ic: meanFinite(ic, lo, hi), t: neweyWestT(ic, lo, hi, mk.meta.horizon) }));
    const lo = mk.windows[0][0], hi = mk.windows[mk.windows.length - 1][1];
    let cover = 0, tot = 0;
    for (let t = lo; t < hi; t++) for (let i = 0; i < mk.N; i++) { tot++; if (Number.isFinite(sig[t * mk.N + i])) cover++; }
    return { daily: ic, windows: wins, mean: wins.reduce((p, w) => p + w.ic, 0) / wins.length, coverage: cover / tot };
  }

  const api = { FEATURES, CONSTANTS, WINDOWS, OPS, MAX_LEN, TOKENS, BY_KEY, Builder, Market, parse, show, evaluate, dailyIC, score, meanFinite };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.PA = api;
})(typeof self !== 'undefined' ? self : this);
