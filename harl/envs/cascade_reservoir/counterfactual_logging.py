"""Exclusive, streamed evidence for frozen model-assisted actor credit."""
import csv
import gzip
from pathlib import Path
import numpy as np
from harl.envs.cascade_reservoir.data_loader import IDS

COUNTERFACTUAL_TRACE_FIELDS = (
    'seed', 'update', 'episode', 'parallel_env_index', 'episode_step', 'date',
    'reservoir_id', 'controllable', 'sampled_action_index', 'joint_action_indices',
    'system_reward', 'team_gae_advantage', 'q_selected', 'counterfactual_baseline',
    'counterfactual_raw_advantage', 'counterfactual_training_advantage', 'advantage_scale',
    'best_action_index', 'q_best', 'best_q_delta', 'selected_action_probability',
    'best_action_probability', 'old_policy_probabilities', 'frozen_q_values',
)


class CounterfactualCreditWriter:
    """Retain candidate values and the exact actor credit before optimization."""
    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.writer = None
        self.rows_written = 0
        self.last_update = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = gzip.open(self.path, 'xt', encoding='utf-8', newline='')
        self.writer = csv.DictWriter(self.stream, fieldnames=COUNTERFACTUAL_TRACE_FIELDS,
                                     lineterminator='\n')
        self.writer.writeheader()
        self.stream.flush()
        return self

    def __exit__(self, *unused):
        if self.stream is not None:
            self.stream.close()
        self.stream = self.writer = None

    def write_batch(self, runner):
        if self.stream is None or self.stream.closed:
            raise RuntimeError('Counterfactual trace writer must be used in its context manager')
        update = runner.update_count+1
        if update <= self.last_update:
            raise ValueError('Counterfactual trace updates must increase')
        credit, context = runner.counterfactual_credit, runner.counterfactual_context
        probability = runner.counterfactual_probabilities
        cb = runner.critic_buffer
        prediction = cb.value_preds[:-1]
        if runner.value_normalizer is not None:
            prediction = runner.value_normalizer.denormalize(prediction)
        team_advantage = cb.returns[:-1,...,0]-prediction[...,0]
        for t in range(runner.length):
            for lane in range(runner.n_envs):
                for i, reservoir in enumerate(IDS):
                    selected = int(context['actions'][t,lane,i])
                    best = int(credit['best_action_indices'][t,lane,i])
                    q = credit['q_values'][t,lane,i]
                    p = probability[t,lane,i]
                    row = dict(seed=runner.seed, update=update,
                        episode=context['episode_ids'][lane], parallel_env_index=lane,
                        episode_step=t+1, date=context['dates'][t,lane], reservoir_id=reservoir,
                        controllable=int(context['controls'][t,lane,i] > 0),
                        sampled_action_index=selected,
                        joint_action_indices='['+','.join(map(str,context['actions'][t,lane]))+']',
                        system_reward=float(cb.rewards[t,lane,0]),
                        team_gae_advantage=float(team_advantage[t,lane]),
                        q_selected=float(credit['q_actual'][t,lane,i]),
                        counterfactual_baseline=float(credit['baseline'][t,lane,i]),
                        counterfactual_raw_advantage=float(credit['raw_advantages'][t,lane,i]),
                        counterfactual_training_advantage=float(runner.counterfactual_advantages[t,lane,i]),
                        advantage_scale=float(credit['advantage_scale']),
                        best_action_index=best, q_best=float(q[best]),
                        best_q_delta=float(q[best]-q[selected]),
                        selected_action_probability=float(p[selected]),
                        best_action_probability=float(p[best]),
                        old_policy_probabilities='['+','.join(format(float(x),'.8g') for x in p)+']',
                        frozen_q_values='['+','.join(format(float(x),'.12g') for x in q)+']')
                    if any(isinstance(value,float) and not np.isfinite(value) for value in row.values()):
                        raise FloatingPointError('Non-finite counterfactual trace value')
                    self.writer.writerow(row)
                    self.rows_written += 1
                self.stream.flush()
        self.last_update = update
