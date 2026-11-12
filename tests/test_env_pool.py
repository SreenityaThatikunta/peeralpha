import numpy as np
import pytest
import torch

from peeralpha.env import AlphaEnv
from peeralpha.metrics import daily_ic, daily_rank_ic, zscore_rows
from peeralpha.pool import AlphaPool
from peeralpha.vocab import Vocab


def test_daily_ic_matches_numpy():
    torch.manual_seed(0)
    a, y = torch.randn(30, 60), torch.randn(30, 60)
    ic = daily_ic(a, y)
    ref = np.array([np.corrcoef(a[i].numpy(), y[i].numpy())[0, 1] for i in range(30)])
    assert np.allclose(ic.numpy(), ref, atol=1e-4)


def test_ic_handles_nan_and_perfect_signal():
    y = torch.randn(10, 50)
    a = y.clone()
    a[:, :5] = float("nan")
    ic = daily_ic(a, y)
    assert torch.allclose(ic, torch.ones(10), atol=1e-5)
    assert torch.allclose(daily_rank_ic(a, y), torch.ones(10), atol=1e-5)


def test_zscore_rows():
    z = zscore_rows(torch.randn(5, 40) * 3 + 7)
    assert torch.allclose(z.mean(1), torch.zeros(5), atol=1e-5)
    assert torch.allclose(z.std(1, unbiased=False), torch.ones(5), atol=1e-4)
    assert torch.isnan(zscore_rows(torch.ones(3, 40))).all()      # no variance -> NaN


def _env(small_market, small_cfg):
    vocab = Vocab(small_cfg.graphs)
    pool = AlphaPool(small_market, small_cfg, vocab)
    return AlphaEnv(small_market, pool, vocab, small_cfg), pool, vocab


def test_env_episode_and_reward(small_market, small_cfg):
    env, pool, vocab = _env(small_market, small_cfg)
    rng = np.random.default_rng(0)
    seen_accept = False
    for _ in range(60):
        obs, _ = env.reset()
        done, steps = False, 0
        while not done:
            a = int(rng.choice(np.flatnonzero(env.action_masks())))
            obs, r, done, trunc, info = env.step(a)
            steps += 1
            assert steps <= small_cfg.max_len
            assert np.isfinite(r)
            assert obs["tokens"].shape == (small_cfg.max_len,)
            assert env.observation_space.contains(obs)
        assert info["status"] in ("invalid", "unstable", "duplicate", "rejected", "accepted", "repeat")
        assert info["n_tokens"] >= min(small_cfg.min_tokens, small_cfg.max_len)
        seen_accept |= info["status"] == "accepted"
    assert seen_accept
    assert pool.size <= pool.capacity


def test_pool_objective_matches_reevaluation(small_market, small_cfg):
    env, pool, vocab = _env(small_market, small_cfg)
    rng = np.random.default_rng(3)
    for _ in range(80):
        env.reset()
        done = False
        while not done:
            _, _, done, _, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
    # the incrementally maintained Gram/weights/objective equal ones recomputed from scratch
    w_inc, obj_inc = pool.w.clone(), pool.obj
    pool._refresh()
    assert torch.allclose(pool.w, w_inc, rtol=1e-3, atol=1e-6)
    assert abs(pool._objective(pool.ensemble()) - obj_inc) < 1e-5


def test_pool_save_load_roundtrip(small_market, small_cfg, tmp_path):
    env, pool, vocab = _env(small_market, small_cfg)
    rng = np.random.default_rng(5)
    for _ in range(60):
        env.reset()
        done = False
        while not done:
            _, _, done, _, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
    p = str(tmp_path / "pool.json")
    pool.save(p)
    loaded = AlphaPool.load(small_market, small_cfg, vocab, p)
    assert loaded.size == pool.size
    assert abs(loaded.obj - pool.obj) < 1e-4
    assert abs(loaded.evaluate_split("valid")["ic"] - pool.evaluate_split("valid")["ic"]) < 1e-5


def test_duplicate_is_penalised(small_market, small_cfg):
    from peeralpha.grammar import build_from_keys
    vocab = Vocab(small_cfg.graphs)
    pool = AlphaPool(small_market, small_cfg, vocab)
    node = build_from_keys(vocab, ["feat:close", "win:5", "Delta"])
    first = pool.try_add(node, 3, 15)
    again = pool.try_add(node, 3, 15)
    assert first.status == "accepted" and again.status == "duplicate"
    assert again.reward < 0


def test_ic_weighting_still_supported(small_market, small_cfg):
    from dataclasses import replace
    cfg = replace(small_cfg, weighting="ic")
    env, pool, vocab = _env(small_market, cfg)
    rng = np.random.default_rng(7)
    for _ in range(80):
        env.reset()
        done = False
        while not done:
            _, _, done, _, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
    assert torch.allclose(pool.w, pool._weights([m.ic for m in pool.members]))
    assert abs(pool._objective(pool.ensemble()) - pool.obj) < 1e-5


def test_repeated_formula_gets_flat_penalty(small_market, small_cfg):
    env, pool, vocab = _env(small_market, small_cfg)
    keys = ["feat:close", "win:5", "Delta"]
    infos = []
    for _ in range(2):
        env.reset()
        for k in keys:
            _, r, done, _, info = env.step(vocab.lookup(k).id)
        _, r, done, _, info = env.step(vocab.sep_id)
        infos.append((r, info["status"]))
    assert infos[0][1] == "accepted"
    assert infos[1] == (-small_cfg.repeat_penalty, "repeat")


def test_longshort_backtest_is_market_neutral():
    from peeralpha.backtest import longshort_backtest
    torch.manual_seed(0)
    ret = torch.randn(120, 50) * 0.01 + 0.002          # common drift must not show up in L/S
    sig = torch.randn(120, 50)
    out = longshort_backtest(sig, ret + 0.05, 10, 120, k=10, rebalance=5, cost=0.0)
    ref = longshort_backtest(sig, ret, 10, 120, k=10, rebalance=5, cost=0.0)
    assert np.allclose(out["port"], ref["port"], atol=1e-6)
    perfect = longshort_backtest(torch.roll(ret, -1, 0), ret, 10, 119, k=10, rebalance=1, cost=0.0)
    assert perfect["annual_return"] > 0


def test_paired_table_and_window_names():
    import evaluate
    assert evaluate._window_and_method("v2nse_w1_v2_graph_all") == ("w1", "v2_graph_all")
    assert evaluate._window_and_method("syn300k_nograph") == ("", "syn300k_nograph")
    groups = {"base": [{"seed": s, "window": "", "test_ic": 0.01} for s in range(3)],
              "new": [{"seed": s, "window": "", "test_ic": 0.02 + 0.001 * s} for s in range(3)]}
    table = evaluate.paired_table(groups, "base")
    assert "| new | 3 | +0.0110 |" in table and "3/3" in table


def _fill(env, seed, n=80):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        env.reset()
        done = False
        while not done:
            _, _, done, _, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))


def test_oof_folds_are_purged(small_market, small_cfg):
    T, H = 300, small_cfg.horizon
    M, fold_of = AlphaPool._part_masks(T, 3, H, "cpu")
    assert M.shape == (T, 4) and bool((M[:, 0] == 1).all())
    assert torch.equal(M[:, 1:].sum(1), (fold_of < 3).float())
    for f in range(2):
        last = int(torch.nonzero(fold_of == f).max())
        first_next = int(torch.nonzero(fold_of == f + 1).min())
        assert first_next - last > H                      # no target window crosses into the next fold


def test_oof_objective_matches_brute_force(small_market, small_cfg):
    from dataclasses import replace
    from peeralpha.metrics import nanmean
    cfg = replace(small_cfg, oof_folds=3)
    env, pool, vocab = _env(small_market, cfg)
    _fill(env, 11)
    assert pool.size >= 2
    obj_inc, w_inc = pool.obj, pool.w.clone()
    pool._refresh()
    Z = torch.stack([m.z for m in pool.members])
    assert torch.allclose(pool.w, w_inc, rtol=1e-3, atol=1e-6)
    assert abs(pool._score(pool.w, pool._G, pool._b, Z) - obj_inc) < 1e-5
    # brute force: ridge on the other folds' days, scored on the held-out fold's days
    X = Z.permute(1, 2, 0) * pool._ymask.unsqueeze(-1)      # [T, N, k]
    ens = torch.zeros_like(pool.y)
    for f in range(3):
        tr = (pool._fold_of != f) & (pool._fold_of < 3)
        Xt, yt = X[tr].reshape(-1, X.shape[-1]), pool._yz[tr].reshape(-1)
        G = Xt.T @ Xt
        lam = cfg.ridge_lambda * float(G.diagonal().mean())
        w = torch.linalg.solve(G + lam * torch.eye(G.shape[0]), Xt.T @ yt)
        ens[pool._fold_of == f] = torch.einsum("k,ktn->tn", w, Z[:, pool._fold_of == f])
    assert abs(nanmean(daily_ic(ens, pool.y)) - obj_inc) < 1e-4


def test_oof_pool_save_load_roundtrip(small_market, small_cfg, tmp_path):
    from dataclasses import replace
    cfg = replace(small_cfg, oof_folds=2)
    env, pool, vocab = _env(small_market, cfg)
    _fill(env, 13, 60)
    p = str(tmp_path / "pool.json")
    pool.save(p)
    loaded = AlphaPool.load(small_market, cfg, vocab, p)
    assert loaded.size == pool.size and abs(loaded.obj - pool.obj) < 1e-4


def test_stability_gate(small_market, small_cfg):
    from dataclasses import replace
    pool = AlphaPool(small_market, replace(small_cfg, stable_folds=4, stable_min_frac=0.75), Vocab(small_cfg.graphs))
    steady = torch.full((400,), 0.02)
    flip = torch.cat([torch.full((200,), 0.05), torch.full((200,), -0.01)])
    assert pool._stable(steady, 0.02)
    assert not pool._stable(flip, 0.02)                    # only 2 of 4 folds agree


def test_unstable_candidate_is_penalised(small_market, small_cfg):
    from dataclasses import replace
    from peeralpha.grammar import build_from_keys
    cfg = replace(small_cfg, stable_folds=4, stable_min_frac=1.0)
    vocab = Vocab(cfg.graphs)
    pool = AlphaPool(small_market, cfg, vocab)
    statuses = set()
    for keys in (["feat:close", "win:5", "Delta"], ["feat:volume", "win:10", "Std"], ["feat:ret", "win:20", "Mean"],
                 ["feat:high", "feat:low", "Sub"], ["feat:vrel", "win:5", "Max"]):
        res = pool.try_add(build_from_keys(vocab, keys), 3, 15)
        statuses.add(res.status)
        if res.status == "unstable":
            assert res.reward == -cfg.unstable_penalty
    assert pool.size == sum(1 for _ in pool.members)
    assert statuses <= {"accepted", "unstable", "duplicate", "rejected"}


def test_newey_west_t():
    from peeralpha.metrics import newey_west_t
    torch.manual_seed(0)
    iid = torch.randn(5000) * 0.1 + 0.01
    plain = float(iid.mean() / (iid.std(unbiased=False) / 5000 ** 0.5))
    assert abs(newey_west_t(iid, 0) - plain) < 1e-3
    assert abs(newey_west_t(iid, 10) - plain) / plain < 0.15
    # a 20-day moving average of noise is autocorrelated: NW must shrink the naive t a lot
    ma = torch.nn.functional.avg_pool1d(iid.view(1, 1, -1), 20, 1).view(-1)
    naive = float(ma.mean() / (ma.std(unbiased=False) / ma.numel() ** 0.5))
    assert newey_west_t(ma, 20) < 0.5 * naive


def test_swap_needs_gain_keeps_objective_monotone(small_market, small_cfg):
    from dataclasses import replace
    cfg = replace(small_cfg, oof_folds=2, swap_needs_gain=True)
    env, pool, vocab = _env(small_market, cfg)
    rng = np.random.default_rng(17)
    objs = []
    for _ in range(120):
        env.reset()
        done = False
        while not done:
            _, _, done, _, info = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
        if pool.size == pool.capacity:
            objs.append(pool.obj)
    assert len(objs) > 10 and all(b >= a - 1e-9 for a, b in zip(objs, objs[1:]))


def test_long_horizon_features(small_market):
    from peeralpha.ops import FEATURES
    close = small_market.feat[FEATURES.index("close")]
    mom = small_market.feat[FEATURES.index("mom12_1")]
    t = 300
    expect = close[t - 20] / close[t - 250] - 1.0
    assert torch.allclose(mom[t], expect, atol=1e-5, equal_nan=True)
    assert torch.isnan(mom[:250]).all()
    vol = small_market.feat[FEATURES.index("vol20")]
    assert torch.isfinite(vol[30:]).float().mean() > 0.9


def test_vocab_windows_and_ops():
    v = Vocab(["sector"], ["ret", "mom12_1"], windows=[5, 120], ops=["Add", "Mean", "NbrMean"])
    assert {t.name for t in v.tokens if t.kind == "win"} == {"5", "120"}
    assert {t.name for t in v.tokens if t.kind in ("unary", "binary", "rolling", "pair", "gop")} == {"Add", "Mean", "NbrMean"}
    with pytest.raises(KeyError):
        Vocab([], ops=["Nope"])


def test_prior_members_are_fixed(small_market, small_cfg, tmp_path):
    from dataclasses import replace
    from peeralpha.grammar import build_from_keys
    cfg = replace(small_cfg, prior="hand_sector", pool_capacity=2, features=["ret", "ret5", "ret20", "vrel"])
    vocab = Vocab(cfg.graphs, cfg.features)
    pool = AlphaPool(small_market, cfg, vocab)
    assert pool.n_fixed == 2 and pool.size == 2 and pool.obj != 0.0
    prior = [str(m.node) for m in pool.members]
    for keys in (["feat:ret", "win:20", "Mean"], ["feat:vrel", "win:5", "Max"], ["feat:ret5", "win:10", "Std"],
                 ["feat:ret20", "CsRank"], ["feat:ret", "win:5", "Sum"], ["feat:vrel", "win:20", "Mean"]):
        if pool.try_add(build_from_keys(vocab, keys), 3, 15).status == "accepted":
            pool.set_last_keys(keys)
    assert [str(m.node) for m in pool.members[:2]] == prior          # never dropped
    assert pool.size <= cfg.pool_capacity + 2
    p = str(tmp_path / "pool.json")
    pool.save(p)
    loaded = AlphaPool.load(small_market, cfg, vocab, p)
    assert loaded.size == pool.size and loaded.n_fixed == 2
    assert abs(loaded.obj - pool.obj) < 1e-4
    # a near-copy of a prior member is a duplicate
    res = pool.try_add(build_from_keys(vocab, ["feat:ret20", "graph:sector", "NbrMean"]), 3, 15)
    assert res.status == "duplicate"


def test_hand_sector_mom_prior(small_market, small_cfg):
    from dataclasses import replace
    cfg = replace(small_cfg, prior="hand_sector_mom", features=["ret", "ret5", "ret20", "mom12_1"])
    pool = AlphaPool(small_market, cfg, Vocab(cfg.graphs, cfg.features))
    assert pool.n_fixed == 3 and [m.keys[-1] for m in pool.members] == ["NbrMean", "NbrRel", "feat:mom12_1"]
