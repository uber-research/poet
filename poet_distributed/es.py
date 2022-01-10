# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import time
import numpy as np
from collections import namedtuple
from .stats import compute_centered_ranks, batched_weighted_sum
from .logger import CSVLogger
import json
import functools

StepStats = namedtuple('StepStats', [
    'po_returns_mean',
    'po_returns_median',
    'po_returns_std',
    'po_returns_max',
    'po_theta_max',
    'po_returns_min',
    'po_len_mean',
    'po_len_std',
    'noise_std',
    'learning_rate',
    'theta_norm',
    'grad_norm',
    'update_ratio',
    'episodes_this_step',
    'timesteps_this_step',
    'time_elapsed_this_step',
])

EvalStats = namedtuple('StepStats', [
    'eval_returns_mean',
    'eval_returns_median',
    'eval_returns_std',
    'eval_len_mean',
    'eval_len_std',
    'eval_n_episodes',
    'time_elapsed',
])

POResult = namedtuple('POResult', [
    'noise_inds',
    'returns',
    'lengths',
])
EvalResult = namedtuple('EvalResult', ['returns', 'lengths'])

logger = logging.getLogger(__name__)


def initialize_master_fiber():
    global noise
    from .noise_module import noise

def initialize_worker_fiber(arg_thetas, arg_niches):
    global noise, thetas, niches
    from .noise_module import noise
    thetas = arg_thetas
    niches = arg_niches

@functools.lru_cache(maxsize=1000)
def fiber_get_theta(iteration, optim_id):
    return thetas[optim_id]

@functools.lru_cache(maxsize=1000)
def fiber_get_niche(iteration, optim_id):
    return niches[optim_id]

def run_eval_batch_fiber(iteration, optim_id, batch_size, rs_seed):
    global noise, niches, thetas
    random_state = np.random.RandomState(rs_seed)
    niche = fiber_get_niche(iteration, optim_id)
    theta = fiber_get_theta(iteration, optim_id)

    returns, lengths = niche.rollout_batch((theta for i in range(batch_size)),
                                           batch_size, random_state, eval=True)

    return EvalResult(returns=returns, lengths=lengths)

def run_po_batch_fiber(iteration, optim_id, batch_size, rs_seed, noise_std):
    global noise, niches, thetas
    random_state = np.random.RandomState(rs_seed)
    niche = fiber_get_niche(iteration, optim_id)
    theta = fiber_get_theta(iteration, optim_id)
    noise_inds = np.asarray([noise.sample_index(random_state, len(theta))
                             for i in range(batch_size)],
                            dtype='int')

    returns = np.zeros((batch_size, 2))
    lengths = np.zeros((batch_size, 2), dtype='int')

    returns[:, 0], lengths[:, 0] = niche.rollout_batch(
        (theta + noise_std * noise.get(noise_idx, len(theta))
         for noise_idx in noise_inds), batch_size, random_state)

    returns[:, 1], lengths[:, 1] = niche.rollout_batch(
        (theta - noise_std * noise.get(noise_idx, len(theta))
         for noise_idx in noise_inds), batch_size, random_state)

    return POResult(returns=returns, noise_inds=noise_inds, lengths=lengths)


class ESOptimizer:
    def __init__(self,
                 fiber_pool,
                 fiber_shared,
                 theta,
                 make_niche,
                 learning_rate,
                 batches_per_chunk,
                 batch_size,
                 eval_batch_size,
                 eval_batches_per_step,
                 l2_coeff,
                 noise_std,
                 lr_decay=1,
                 lr_limit=0.001,
                 noise_decay=1,
                 noise_limit=0.01,
                 normalize_grads_by_noise_std=False,
                 returns_normalization='centered_ranks',
                 optim_id=0,
                 log_file='unname.log',
                 created_at=0,
                 is_candidate=False):

        from .optimizers import Adam, SimpleSGD

        logger.debug('Creating optimizer {}...'.format(optim_id))
        self.fiber_pool = fiber_pool
        self.fiber_shared = fiber_shared

        self.optim_id = optim_id
        assert self.fiber_pool is not None

        self.theta = theta
        #print(self.theta)
        logger.debug('Optimizer {} optimizing {} parameters'.format(
            optim_id, len(theta)))
        self.optimizer = Adam(self.theta, stepsize=learning_rate)
        self.sgd_optimizer = SimpleSGD(stepsize=learning_rate)
        self.lr_decay = lr_decay
        self.lr_limit = lr_limit
        self.noise_decay = noise_decay
        self.noise_limit = noise_limit

        self.fiber_shared = fiber_shared
        niches = fiber_shared["niches"]
        niches[optim_id] = make_niche()

        self.batches_per_chunk = batches_per_chunk
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.eval_batches_per_step = eval_batches_per_step
        self.l2_coeff = l2_coeff
        self.noise_std = noise_std
        self.init_noise_std = noise_std

        self.normalize_grads_by_noise_std = normalize_grads_by_noise_std
        self.returns_normalization = returns_normalization

        if is_candidate == False:
            log_fields = [
                'po_returns_mean_{}'.format(optim_id),
                'po_returns_median_{}'.format(optim_id),
                'po_returns_std_{}'.format(optim_id),
                'po_returns_max_{}'.format(optim_id),
                'po_returns_min_{}'.format(optim_id),
                'po_len_mean_{}'.format(optim_id),
                'po_len_std_{}'.format(optim_id),
                'noise_std_{}'.format(optim_id),
                'learning_rate_{}'.format(optim_id),
                'eval_returns_mean_{}'.format(optim_id),
                'eval_returns_median_{}'.format(optim_id),
                'eval_returns_std_{}'.format(optim_id),
                'eval_len_mean_{}'.format(optim_id),
                'eval_len_std_{}'.format(optim_id),
                'eval_n_episodes_{}'.format(optim_id),
                'theta_norm_{}'.format(optim_id),
                'grad_norm_{}'.format(optim_id),
                'update_ratio_{}'.format(optim_id),
                'episodes_this_step_{}'.format(optim_id),
                'episodes_so_far_{}'.format(optim_id),
                'timesteps_this_step_{}'.format(optim_id),
                'timesteps_so_far_{}'.format(optim_id),
                'time_elapsed_this_step_{}'.format(optim_id),

                'accept_theta_in_{}'.format(optim_id),
                'eval_returns_mean_best_in_{}'.format(optim_id),
                'eval_returns_mean_best_with_ckpt_in_{}'.format(optim_id),
                'eval_returns_mean_theta_from_others_in_{}'.format(optim_id),
                'eval_returns_mean_proposal_from_others_in_{}'.format(optim_id),
            ]
            log_path = log_file + '/' + log_file.split('/')[-1] + '.' + optim_id + '.log'
            self.data_logger = CSVLogger(log_path, log_fields + [
                'time_elapsed_so_far',
                'iteration',
            ])
            logger.info('Optimizer {} created!'.format(optim_id))

        self.filename_best = log_file + '/' + log_file.split('/')[-1] + '.' + optim_id + '.best.json'
        self.log_data = {}
        self.t_start = time.time()
        self.episodes_so_far = 0
        self.timesteps_so_far = 0

        self.checkpoint_thetas = None
        self.checkpoint_scores = None

        self.self_evals = None   # Score of current parent theta
        self.proposal = None   # Score of best transfer
        self.proposal_theta = None # Theta of best transfer
        self.proposal_source = None # Source of best transfer

        self.created_at = created_at
        self.start_score = None

        self.best_score = None
        self.best_theta = None

        self.iteration = 0

    def __del__(self):
        logger.debug('Optimizer {} cleanning up workers...'.format(
            self.optim_id))

    def clean_dicts_before_iter(self):
        self.log_data.clear()
        self.self_evals = None
        self.proposal = None
        self.proposal_theta = None
        self.proposal_source = None

    def pick_proposal(self, checkpointing, reset_optimizer):

        accept_key = 'accept_theta_in_{}'.format(
                self.optim_id)
        if checkpointing and self.checkpoint_scores > self.proposal:
            self.log_data[accept_key] = 'do_not_consider_CP'
        else:
            self.log_data[accept_key] = '{}'.format(
                self.proposal_source)
            if self.optim_id != self.proposal_source:
                self.set_theta(
                    self.proposal_theta,
                    reset_optimizer=reset_optimizer)
                self.self_evals = self.proposal

        self.checkpoint_thetas = np.array(self.theta)
        self.checkpoint_scores = self.self_evals

        if self.best_score < self.self_evals:
            self.best_score = self.self_evals
            self.best_theta = np.array(self.theta)

    def save_to_logger(self, iteration):
        self.log_data['time_elapsed_so_far'] = time.time() - self.t_start
        self.log_data['iteration'] = iteration
        self.data_logger.log(**self.log_data)

        logger.debug('iter={} Optimizer {} best score {}'.format(
            iteration, self.optim_id, self.best_score))

        #if iteration % 100 == 0:
        #    self.save_policy(self.filename_best+'.arxiv.'+str(iteration))

        self.save_policy(self.filename_best)

    def save_policy(self, policy_file, reset=False):
        if self.best_score is not None and self.best_theta is not None:
            with open(policy_file, 'wt') as out:
                json.dump([self.best_theta.tolist(), self.best_score], out, sort_keys=True, indent=0, separators=(',', ': '))
            if reset:
                self.best_score = None
                self.best_theta = None


    def update_dicts_after_transfer(self, source_optim_id, source_optim_theta, stats, keyword):
        eval_key = 'eval_returns_mean_{}_from_others_in_{}'.format(keyword,  # noqa
            self.optim_id)
        if eval_key not in self.log_data.keys():
            self.log_data[eval_key] = source_optim_id + '_' + str(stats.eval_returns_mean)
        else:
            self.log_data[eval_key] += '_' + source_optim_id + '_' + str(stats.eval_returns_mean)

        if stats.eval_returns_mean > self.proposal:
            self.proposal = stats.eval_returns_mean
            self.proposal_source = source_optim_id + ('' if keyword=='theta' else "_proposal")
            self.proposal_theta = np.array(source_optim_theta)

    def update_dicts_after_es(self, stats, self_eval_stats):

        self.self_evals = self_eval_stats.eval_returns_mean
        if self.start_score is None:
            self.start_score = self.self_evals
        self.proposal = self_eval_stats.eval_returns_mean
        self.proposal_source = self.optim_id
        self.proposal_theta = np.array(self.theta)

        if self.checkpoint_scores is None:
            self.checkpoint_thetas = np.array(self.theta)
            self.checkpoint_scores = self_eval_stats.eval_returns_mean

        self.episodes_so_far += stats.episodes_this_step
        self.timesteps_so_far += stats.timesteps_this_step

        if self.best_score is None or self.best_score < self.self_evals:
            self.best_score = self.self_evals
            self.best_theta = np.array(self.theta)

        self.log_data.update({
            'po_returns_mean_{}'.format(self.optim_id):
                stats.po_returns_mean,
            'po_returns_median_{}'.format(self.optim_id):
                stats.po_returns_median,
            'po_returns_std_{}'.format(self.optim_id):
                stats.po_returns_std,
            'po_returns_max_{}'.format(self.optim_id):
                stats.po_returns_max,
            'po_returns_min_{}'.format(self.optim_id):
                stats.po_returns_min,
            'po_len_mean_{}'.format(self.optim_id):
                stats.po_len_mean,
            'po_len_std_{}'.format(self.optim_id):
                stats.po_len_std,
            'noise_std_{}'.format(self.optim_id):
                stats.noise_std,
            'learning_rate_{}'.format(self.optim_id):
                stats.learning_rate,
            'eval_returns_mean_{}'.format(self.optim_id):
                self_eval_stats.eval_returns_mean,
            'eval_returns_median_{}'.format(self.optim_id):
                self_eval_stats.eval_returns_median,
            'eval_returns_std_{}'.format(self.optim_id):
                self_eval_stats.eval_returns_std,
            'eval_len_mean_{}'.format(self.optim_id):
                self_eval_stats.eval_len_mean,
            'eval_len_std_{}'.format(self.optim_id):
                self_eval_stats.eval_len_std,
            'eval_n_episodes_{}'.format(self.optim_id):
                self_eval_stats.eval_n_episodes,
            'theta_norm_{}'.format(self.optim_id):
                stats.theta_norm,
            'grad_norm_{}'.format(self.optim_id):
                stats.grad_norm,
            'update_ratio_{}'.format(self.optim_id):
                stats.update_ratio,
            'episodes_this_step_{}'.format(self.optim_id):
                stats.episodes_this_step,
            'episodes_so_far_{}'.format(self.optim_id):
                self.episodes_so_far,
            'timesteps_this_step_{}'.format(self.optim_id):
                stats.timesteps_this_step,
            'timesteps_so_far_{}'.format(self.optim_id):
                self.timesteps_so_far,
            'time_elapsed_this_step_{}'.format(self.optim_id):
                stats.time_elapsed_this_step + self_eval_stats.time_elapsed,
            'accept_theta_in_{}'.format(self.optim_id): 'self'
        })


    def broadcast_theta(self, theta):
        '''On all worker, set thetas[this optimizer] to theta'''
        logger.debug('Optimizer {} broadcasting theta...'.format(self.optim_id))

        thetas = self.fiber_shared["thetas"]
        thetas[self.optim_id] = theta
        self.iteration += 1


    def add_env(self, env):
        '''On all worker, add env_name to niche'''
        logger.debug('Optimizer {} add env {}...'.format(self.optim_id, env.name))

        thetas = self.fiber_shared["niches"]
        niches[self.optim_id].add_env(env)

    def delete_env(self, env_name):
        '''On all worker, delete env from niche'''
        logger.debug('Optimizer {} delete env {}...'.format(self.optim_id, env_name))

        niches = self.fiber_shared["niches"]
        niches[self.optim_id].delete_env(env_name)

    def start_chunk_fiber(self, runner, batches_per_chunk, batch_size, *args):
        logger.debug('Optimizer {} spawning {} batches of size {}'.format(
            self.optim_id, batches_per_chunk, batch_size))

        rs_seeds = np.random.randint(np.int32(2 ** 31 - 1), size=batches_per_chunk)

        chunk_tasks = []
        pool = self.fiber_pool
        niches = self.fiber_shared["niches"]
        thetas = self.fiber_shared["thetas"]

        for i in range(batches_per_chunk):
            chunk_tasks.append(
                pool.apply_async(runner, args=(self.iteration,
                    self.optim_id, batch_size, rs_seeds[i])+args))
        return chunk_tasks

    def get_chunk(self, tasks):
        return [task.get() for task in tasks]

    def collect_po_results(self, po_results):
        noise_inds = np.concatenate([r.noise_inds for r in po_results])
        returns = np.concatenate([r.returns for r in po_results])
        lengths = np.concatenate([r.lengths for r in po_results])
        return noise_inds, returns, lengths

    def collect_eval_results(self, eval_results):
        eval_returns = np.concatenate([r.returns for r in eval_results])
        eval_lengths = np.concatenate([r.lengths for r in eval_results])
        return eval_returns, eval_lengths

    def compute_grads(self, step_results, theta):
        noise_inds, returns, _ = self.collect_po_results(step_results)

        pos_row, neg_row = returns.argmax(axis=0)
        noise_sign = 1.0
        po_noise_ind_max = noise_inds[pos_row]

        if returns[pos_row, 0] < returns[neg_row, 1]:
            noise_sign = -1.0
            po_noise_ind_max = noise_inds[neg_row]

        po_theta_max = theta + noise_sign * self.noise_std * noise.get(po_noise_ind_max, len(theta))

        if self.returns_normalization == 'centered_ranks':
            proc_returns = compute_centered_ranks(returns)
        elif self.returns_normalization == 'normal':
            proc_returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        else:
            raise NotImplementedError(
                'Invalid return normalization `{}`'.format(
                    self.returns_normalization))
        grads, _ = batched_weighted_sum(
            proc_returns[:, 0] - proc_returns[:, 1],
            (noise.get(idx, len(theta)) for idx in noise_inds),
            batch_size=500)

        grads /= len(returns)
        if self.normalize_grads_by_noise_std:
            grads /= self.noise_std
        return grads, po_theta_max

    def set_theta(self, theta, reset_optimizer=True):
        self.theta = np.array(theta)
        if reset_optimizer:
            self.optimizer.reset()
            self.noise_std = self.init_noise_std

    def start_theta_eval(self, theta):
        '''eval theta in this optimizer's niche'''
        step_t_start = time.time()
        self.broadcast_theta(theta)

        eval_tasks = self.start_chunk_fiber(
            run_eval_batch_fiber, self.eval_batches_per_step, self.eval_batch_size)

        return eval_tasks, theta, step_t_start

    def get_theta_eval(self, res):
        eval_tasks, theta, step_t_start = res
        eval_results = self.get_chunk(eval_tasks)
        eval_returns, eval_lengths = self.collect_eval_results(eval_results)
        step_t_end = time.time()

        logger.debug(
            'get_theta_eval {} finished running {} episodes, {} timesteps'.format(
                self.optim_id, len(eval_returns), eval_lengths.sum()))

        return EvalStats(
            eval_returns_mean=eval_returns.mean(),
            eval_returns_median=np.median(eval_returns),
            eval_returns_std=eval_returns.std(),
            eval_len_mean=eval_lengths.mean(),
            eval_len_std=eval_lengths.std(),
            eval_n_episodes=len(eval_returns),
            time_elapsed=step_t_end - step_t_start,
        )

    def start_step(self, theta=None):
        ''' based on theta (if none, this optimizer's theta)
            generate the P.O. cloud, and eval them in this optimizer's niche
        '''
        step_t_start = time.time()
        if theta is None:
            theta = self.theta
        self.broadcast_theta(theta)

        step_results = self.start_chunk_fiber(
            run_po_batch_fiber,
            self.batches_per_chunk,
            self.batch_size,
            self.noise_std)

        return step_results, theta, step_t_start

    def get_step(self, res, propose_with_adam=True, decay_noise=True, propose_only=False):
        step_tasks, theta, step_t_start = res
        step_results = self.get_chunk(step_tasks)

        _, po_returns, po_lengths = self.collect_po_results(
            step_results)
        episodes_this_step = len(po_returns)
        timesteps_this_step = po_lengths.sum()

        logger.debug(
            'Optimizer {} finished running {} episodes, {} timesteps'.format(
                self.optim_id, episodes_this_step, timesteps_this_step))

        grads, po_theta_max = self.compute_grads(step_results, theta)
        if not propose_only:
            update_ratio, theta = self.optimizer.update(
                theta, -grads + self.l2_coeff * theta)

            self.optimizer.stepsize = max(
                self.optimizer.stepsize * self.lr_decay, self.lr_limit)
            if decay_noise:
                self.noise_std = max(
                    self.noise_std * self.noise_decay, self.noise_limit)

        else:  #only make proposal
            if propose_with_adam:
                update_ratio, theta = self.optimizer.propose(
                    theta, -grads + self.l2_coeff * theta)
            else:
                update_ratio, theta = self.sgd_optimizer.compute(
                    theta, -grads + self.l2_coeff * theta)  # keeps no state
        logger.debug(
            'Optimizer {} finished computing gradients'.format(
                self.optim_id))

        step_t_end = time.time()

        return theta, StepStats(
            po_returns_mean=po_returns.mean(),
            po_returns_median=np.median(po_returns),
            po_returns_std=po_returns.std(),
            po_returns_max=po_returns.max(),
            po_theta_max=po_theta_max,
            po_returns_min=po_returns.min(),
            po_len_mean=po_lengths.mean(),
            po_len_std=po_lengths.std(),
            noise_std=self.noise_std,
            learning_rate=self.optimizer.stepsize,
            theta_norm=np.square(theta).sum(),
            grad_norm=float(np.square(grads).sum()),
            update_ratio=float(update_ratio),
            episodes_this_step=episodes_this_step,
            timesteps_this_step=timesteps_this_step,
            time_elapsed_this_step=step_t_end - step_t_start,
        )

    def evaluate_theta(self, theta):
        self_eval_task = self.start_theta_eval(theta)
        self_eval_stats = self.get_theta_eval(self_eval_task)
        return self_eval_stats.eval_returns_mean


    def evaluate_transfer(self, optimizers, propose_with_adam=False):

        best_init_score = None
        best_init_theta = None

        for source_optim in optimizers.values():
            score = self.evaluate_theta(source_optim.theta)
            if best_init_score == None or score > best_init_score:
                best_init_score = score
                best_init_theta = np.array(source_optim.theta)

            task = self.start_step(source_optim.theta)
            proposed_theta, _ = self.get_step(
                task, propose_with_adam=propose_with_adam, propose_only=True)
            score = self.evaluate_theta(proposed_theta)
            if score > best_init_score:
                best_init_score = score
                best_init_theta = np.array(proposed_theta)

        return best_init_score, best_init_theta
