import numpy as np
import torch

from graphalpha.env import AlphaEnv
from graphalpha.metrics import daily_ic, daily_rank_ic, zscore_rows
from graphalpha.pool import AlphaPool
from graphalpha.vocab import Vocab


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
        assert info["status"] in ("invalid", "duplicate", "rejected", "accepted", "repeat")
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
    from graphalpha.grammar import build_from_keys
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
    from graphalpha.backtest import longshort_backtest
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
