"""CPU stress tests of the unchanged predictor and original REINFORCE selector."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from hardness_predictor import HardnessPredictor, _features
from package_training_inputs import atomic_json, sha256
from selector_simulation import load_inputs, SimulatedRewards, ROOT, SCENARIOS, METHODS

STATE = ROOT/'modal_results/selector_simulations'
REPORT = ROOT/'reports/selector-simulations'
STEPS = 20000
SEEDS = (43, 44, 45, 46, 47)


def initialize():
    global INPUTS
    torch.set_num_threads(1)
    INPUTS = load_inputs()


def run_cell(task):
    scenario, method, seed = task
    folder = STATE/'cells'/f'{scenario}-{method}-{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    done = folder/'complete.json'
    if done.exists():
        receipt = json.loads(done.read_text())
        if receipt['trace_sha256'] != sha256(folder/'trace.npz'):
            raise ValueError('Completed simulation trace changed')
        return receipt
    configs, rewards, calibration = INPUTS
    env = SimulatedRewards(rewards, configs, calibration, scenario, seed, STEPS)
    learner = (HardnessPredictor(configs, seed=seed) if method == 'predictor' else
               _features.SettingsTokenizerBandit(configs, seed=seed, entropy_weight=0.) if method == 'reinforce' else None)
    rng = np.random.default_rng(seed)
    actions = np.empty(STEPS, dtype=np.int16)
    observations = np.empty(STEPS)
    chosen = np.empty(STEPS)
    oracle = np.empty(STEPS)
    modes = np.zeros(STEPS, dtype=np.uint8)
    top100 = np.empty(STEPS, dtype=np.uint8)
    ranks = np.empty(STEPS, dtype=np.int16)
    started = time.monotonic()
    events = folder/'events.jsonl'
    events.write_text('')
    for offset in range(STEPS):
        step = offset+1
        if method == 'predictor':
            selection = learner.select(); action = selection['index']
            modes[offset] = int(selection['mode'] == 'explore')
        elif method == 'reinforce':
            action = learner.sample()
        else:
            action = int(rng.integers(1000))
        means = env.means(step)
        # Full counterfactual vector is used only by the simulator's evaluator.
        actions[offset] = action; chosen[offset] = means[action]; oracle[offset] = means.max()
        ranks[offset] = 1+int((means > means[action]).sum()); top100[offset] = ranks[offset] <= 100
        reward = env.observe(action, step, means)
        observations[offset] = reward
        if method == 'predictor':
            learner.observe(action, reward)
        elif method == 'reinforce':
            learner.update(action, reward)
        if step % 100 == 0:
            counts = np.bincount(actions[max(0, step-1000):step], minlength=1000)
            record = dict(step=step, scenario=scenario, method=method, seed=seed,
                at_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-started,
                missed_hardness=float((oracle[step-100:step]-chosen[step-100:step]).mean()),
                mean_hardness=float(chosen[step-100:step].mean()), mean_oracle=float(oracle[step-100:step].mean()),
                top100_fraction=float(top100[step-100:step].mean()),
                unique_visited=int(np.unique(actions[:step]).size),
                largest_recent_share=float(counts.max()/counts.sum()))
            atomic_json(folder/'status.json', dict(stage='running', completed=step, **record))
            with events.open('a') as stream:
                stream.write(json.dumps(record)+'\n')
    temporary = folder/'trace.tmp.npz'
    np.savez_compressed(temporary, actions=actions, rewards=observations, chosen_mean=chosen,
        oracle_mean=oracle, exploration=modes, selected_rank=ranks)
    temporary.replace(folder/'trace.npz')
    receipt = dict(scenario=scenario, method=method, seed=seed, steps=STEPS,
        trace_sha256=sha256(folder/'trace.npz'), seconds=time.monotonic()-started,
        mean_missed_hardness=float((oracle-chosen).mean()),
        final_2000_missed_hardness=float((oracle[-2000:]-chosen[-2000:]).mean()),
        final_2000_top100_fraction=float(top100[-2000:].mean()),
        unique_visited=int(np.unique(actions).size))
    atomic_json(done, receipt)
    atomic_json(folder/'status.json', dict(stage='complete', completed=STEPS, **record))
    return receipt


def emit(completed, workers, started, stage='running'):
    elapsed = time.monotonic()-started; total = len(SCENARIOS)*len(METHODS)*len(SEEDS)*STEPS
    record = dict(stage=stage, state='running' if stage=='running' else stage,
        completed=completed, total=total, at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=elapsed, eta_seconds=(total-completed)*elapsed/completed if completed else None,
        platform='Local · CPU only', workers=workers)
    atomic_json(STATE/'status.json', record)
    with (STATE/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    lock = (STATE/'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    _, _, calibration = load_inputs()
    sources = ['reports/regret-pool-ranking-6500/all-scores.json.gz',
        'reports/regret-pool-ranking-6500/tokenizer-ranking.csv',
        'reports/regret-pool-ranking-6500/pool-index.json',
        'reports/regret-favorite-trajectory/learning-curve.csv',
        'scripts/hardness_predictor.py', 'scripts/selector_simulation.py',
        'scripts/run_selector_simulations.py',
        'runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py']
    plan = dict(steps=STEPS, seeds=list(SEEDS), scenarios=list(SCENARIOS), methods=list(METHODS),
        tokenizers=1000, measured_batches_per_tokenizer=32, calibration=calibration,
        learner_initialization='Fresh original initialization for every cell; no ranking labels supplied to learner',
        predictor='Unchanged approved predictor, Adam .001, one-in-four shuffled coverage, single-observation MSE',
        reinforce='Original SettingsTokenizerBandit, Adam .001, EMA .99, entropy 0, sampled softmax',
        noise='Resample the same shared measured batch index per step; action-specific centered residuals unchanged',
        gradual='At 10000: linearly lower every Unigram mean by observed #771 gap drop over 1500 steps',
        abrupt='At 10000: lower every Unigram mean by twice the observed #771 gap drop',
        practice='Exponential mean decay per selection calibrated to #771 6500→8000; tokenizer-only or entire algorithm family',
        interpretation='Stress tests of finite-table bandits; synthetic changes are not inferred LM counterfactuals. Practice environments differ by selector history; compare current-world missed hardness, not absolute reward alone.',
        source_sha256={name:sha256(ROOT/name) for name in sources})
    if (STATE/'plan.json').exists() and json.loads((STATE/'plan.json').read_text()) != plan:
        raise ValueError('Simulation sources/settings changed; use a separate run directory')
    atomic_json(STATE/'plan.json', plan)
    started = time.monotonic()
    emit(0, [], started)
    tasks = [(scenario, method, seed) for scenario in SCENARIOS for seed in SEEDS for method in METHODS]
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        futures = {pool.submit(run_cell, task): task for task in tasks}
        remaining = set(futures)
        while remaining:
            for future in list(remaining):
                if future.done():
                    future.result(); remaining.remove(future)
            completed = 0; workers = []
            for scenario, method, seed in tasks:
                path = STATE/'cells'/f'{scenario}-{method}-{seed}'/'status.json'
                if not path.exists():
                    continue
                status = json.loads(path.read_text()); completed += status['completed']
                if status['stage'] == 'running':
                    workers.append(dict(id=f'{scenario}-{method}-{seed}', profile='local',
                        platform='Local · CPU', state=f"{scenario} · {method} · seed {seed} · {status['completed']:,}/20,000"))
            emit(completed, workers, started)
            if int(time.monotonic()-started)//30 != getattr(main, '_last_report', -1):
                subprocess.run([sys.executable, 'scripts/report_selector_simulations.py'], cwd=ROOT, check=True)
                main._last_report = int(time.monotonic()-started)//30
            if remaining:
                time.sleep(5)
    subprocess.run([sys.executable, 'scripts/report_selector_simulations.py'], cwd=ROOT, check=True)
    emit(len(tasks)*STEPS, [], started, 'complete')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        old = json.loads((STATE/'status.json').read_text()) if (STATE/'status.json').exists() else {}
        old.update(stage='failed', state='failed', error=str(error), workers=[], at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(STATE/'status.json', old)
        raise
