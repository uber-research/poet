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


from .logger import CSVLogger
import logging
logger = logging.getLogger(__name__)
import numpy as np
from poet_distributed.es import ESOptimizer
from poet_distributed.es import initialize_worker_fiber
from collections import OrderedDict
from poet_distributed.niches.box2d.env import Env_config
from poet_distributed.niches.box2d.cppn import CppnEnvParams
from poet_distributed.reproduce_ops import Reproducer
from poet_distributed.novelty import compute_novelty_vs_archive
import json


def construct_niche_fns_from_env(args, env, env_params, seed):
    def niche_wrapper(configs, env_params, seed):  # force python to make a new lexical scope
        def make_niche():
            from poet_distributed.niches import Box2DNiche
            return Box2DNiche(env_configs=configs,
                            env_params=env_params,
                            seed=seed,
                            init=args.init,
                            stochastic=args.stochastic)

        return make_niche

    niche_name = env.name
    configs = (env,)

    return niche_name, niche_wrapper(list(configs), env_params, seed)


class MultiESOptimizer:
    def __init__(self, args):

        self.args = args

        import fiber as mp

        mp_ctx = mp.get_context('spawn')
        manager = mp_ctx.Manager()
        self.manager = manager
        self.fiber_shared = {
                "niches": manager.dict(),
                "thetas": manager.dict(),
        }
        self.fiber_pool = mp_ctx.Pool(args.num_workers, initializer=initialize_worker_fiber,
                initargs=(self.fiber_shared["thetas"],
                    self.fiber_shared["niches"]))

        self.ANNECS = 0
        self.env_registry = OrderedDict()
        self.env_archive = OrderedDict()
        self.env_reproducer = Reproducer(args)
        self.optimizers = OrderedDict()
        self.archived_optimizers = OrderedDict()
       
        env = Env_config(
            name='flat',
            ground_roughness=0,
            pit_gap=[],
            stump_width=[],
            stump_height=[],
            stump_float=[],
            stair_height=[],
            stair_width=[],
            stair_steps=[])

        params = CppnEnvParams()
        self.add_optimizer(env=env, cppn_params=params, seed=args.master_seed)

    def create_optimizer(self, env, cppn_params, seed, created_at=0, model_params=None, is_candidate=False):

        assert env != None
        assert cppn_params != None

        optim_id, niche_fn = construct_niche_fns_from_env(args=self.args, env=env, env_params=cppn_params, seed=seed)

        niche = niche_fn()
        if model_params is not None:
            theta = np.array(model_params)
        else:
            theta=niche.initial_theta()
        assert optim_id not in self.optimizers.keys()

        return ESOptimizer(
            optim_id=optim_id,
            fiber_pool=self.fiber_pool,
            fiber_shared=self.fiber_shared,
            theta=theta,
            make_niche=niche_fn,
            learning_rate=self.args.learning_rate,
            lr_decay=self.args.lr_decay,
            lr_limit=self.args.lr_limit,
            batches_per_chunk=self.args.batches_per_chunk,
            batch_size=self.args.batch_size,
            eval_batch_size=self.args.eval_batch_size,
            eval_batches_per_step=self.args.eval_batches_per_step,
            l2_coeff=self.args.l2_coeff,
            noise_std=self.args.noise_std,
            noise_decay=self.args.noise_decay,
            normalize_grads_by_noise_std=self.args.normalize_grads_by_noise_std,
            returns_normalization=self.args.returns_normalization,
            noise_limit=self.args.noise_limit,
            log_file=self.args.log_file,
            created_at=created_at,
            is_candidate=is_candidate)


    def add_optimizer(self, env, cppn_params, seed, created_at=0, model_params=None):
        '''
            creat a new optimizer/niche
            created_at: the iteration when this niche is created
        '''
        o = self.create_optimizer(env, cppn_params, seed, created_at, model_params)
        optim_id = o.optim_id
        self.optimizers[optim_id] = o

        assert optim_id not in self.env_registry.keys()
        assert optim_id not in self.env_archive.keys()
        self.env_registry[optim_id] = (env, cppn_params)
        self.env_archive[optim_id] = (env, cppn_params)

    def archive_optimizer(self, optim_id):
        assert optim_id in self.optimizers.keys()
        #assume optim_id == env_id for single_env niches
        o = self.optimizers.pop(optim_id)
        assert optim_id in self.env_registry.keys()
        self.env_registry.pop(optim_id)
        logger.info('Archived {} '.format(optim_id))
        self.archived_optimizers[optim_id] = o

    def ind_es_step(self, iteration):
        tasks = [o.start_step() for o in self.optimizers.values()]

        for optimizer, task in zip(self.optimizers.values(), tasks):

            optimizer.theta, stats = optimizer.get_step(task)
            self_eval_task = optimizer.start_theta_eval(optimizer.theta)
            self_eval_stats = optimizer.get_theta_eval(self_eval_task)

            logger.info('Iter={} Optimizer {} theta_mean {} best po {} iteration spent {}'.format(
                iteration, optimizer.optim_id, self_eval_stats.eval_returns_mean,
                stats.po_returns_max, iteration - optimizer.created_at))

            optimizer.update_dicts_after_es(stats=stats,
                self_eval_stats=self_eval_stats)

    def transfer(self, propose_with_adam, checkpointing, reset_optimizer):
        logger.info('Computing direct transfers...')
        proposal_targets = {}
        for source_optim in self.optimizers.values():
            source_tasks = []
            proposal_targets[source_optim] = []
            for target_optim in [o for o in self.optimizers.values()
                                    if o is not source_optim]:
                task = target_optim.start_theta_eval(
                    source_optim.theta)
                source_tasks.append((task, target_optim))

            for task, target_optim in source_tasks:
                stats = target_optim.get_theta_eval(task)

                try_proposal = target_optim.update_dicts_after_transfer(source_optim_id=source_optim.optim_id,
                                    source_optim_theta=source_optim.theta,
                                    stats=stats, keyword='theta')
                if try_proposal:
                    proposal_targets[source_optim].append(target_optim)

        logger.info('Computing proposal transfers...')
        for source_optim in self.optimizers.values():
            source_tasks = []
            for target_optim in [o for o in self.optimizers.values()
                                    if o is not source_optim]:
                if target_optim in proposal_targets[source_optim]:
                    task = target_optim.start_step(source_optim.theta)
                    source_tasks.append((task, target_optim))

            for task, target_optim in source_tasks:
                proposed_theta, _ = target_optim.get_step(
                    task, propose_with_adam=propose_with_adam, propose_only=True)

                proposal_eval_task = target_optim.start_theta_eval(proposed_theta)
                proposal_eval_stats = target_optim.get_theta_eval(proposal_eval_task)

                target_optim.update_dicts_after_transfer(source_optim_id=source_optim.optim_id,
                    source_optim_theta=proposed_theta,
                    stats=proposal_eval_stats, keyword='proposal')

        logger.info('Considering transfers...')
        for o in self.optimizers.values():
            o.pick_proposal(checkpointing, reset_optimizer)

    def check_optimizer_status(self, iteration):
        '''
            return two lists
        '''
        logger.info("health_check")
        repro_candidates, delete_candidates = [], []
        for optim_id in self.env_registry.keys():
            o = self.optimizers[optim_id]
            logger.info("niche {} created at {} start_score {} current_self_evals {}".format(
                optim_id, o.created_at, o.start_score, o.self_evals))
            if o.self_evals >= self.args.repro_threshold:
                repro_candidates.append(optim_id)

        logger.debug("candidates to reproduce")
        logger.debug(repro_candidates)
        logger.debug("candidates to delete")
        logger.debug(delete_candidates)

        return repro_candidates, delete_candidates


    def pass_dedup(self, env_config):
        if env_config.name in self.env_registry.keys():
            logger.debug("active env already. reject!")
            return False
        else:
            return True

    def pass_mc(self, score):
        if score < self.args.mc_lower or score > self.args.mc_upper:
            return False
        else:
            return True

    def get_new_env(self, list_repro):

        optim_id = self.env_reproducer.pick(list_repro)
        assert optim_id in self.optimizers.keys()
        assert optim_id in self.env_registry.keys()
        parent_env_config, parent_cppn_params = self.env_registry[optim_id]
        child_env_config = self.env_reproducer.mutate(parent_env_config, no_mutate=True)
        child_cppn_params = parent_cppn_params.get_mutated_params()

        logger.info("we pick to mutate: {} and we got {} back".format(optim_id, child_env_config.name))
        logger.debug("parent")
        logger.debug(parent_env_config)
        logger.debug("child")
        logger.debug(child_env_config)

        seed = np.random.randint(1000000)
        return child_env_config, child_cppn_params, seed, optim_id

    def get_child_list(self, parent_list, max_children):
        child_list = []

        mutation_trial = 0
        while mutation_trial < max_children:
            new_env_config, new_cppn_params, seed, parent_optim_id = self.get_new_env(parent_list)
            mutation_trial += 1
            if self.pass_dedup(new_env_config):
                o = self.create_optimizer(new_env_config, new_cppn_params, seed, is_candidate=True)
                score = o.evaluate_theta(self.optimizers[parent_optim_id].theta)
                if self.pass_mc(score):
                    novelty_score = compute_novelty_vs_archive(self.archived_optimizers, self.optimizers, o, k=5,
                                        low=self.args.mc_lower, high=self.args.mc_upper)
                    logger.debug("{} passed mc, novelty score {}".format(score, novelty_score))
                    child_list.append((new_env_config, new_cppn_params, seed, parent_optim_id, novelty_score))
                del o

        #sort child list according to novelty for high to low
        child_list = sorted(child_list,key=lambda x: x[4], reverse=True)
        return child_list

    def adjust_envs_niches(self, iteration, steps_before_adjust, max_num_envs=None, max_children=8, max_admitted=1):

        if iteration > 0 and iteration % steps_before_adjust == 0:
            list_repro, list_delete = self.check_optimizer_status(iteration)

            if len(list_repro) == 0:
                return

            logger.info("list of niches to reproduce")
            logger.info(list_repro)
            logger.info("list of niches to delete")
            logger.info(list_delete)

            for optim in self.optimizers.values():
                optim.update_pata_ec(self.archived_optimizers, self.optimizers, self.args.mc_lower, self.args.mc_upper)

            for optim in self.archived_optimizers.values():
                optim.update_pata_ec(self.archived_optimizers, self.optimizers, self.args.mc_lower, self.args.mc_upper)

            child_list = self.get_child_list(list_repro, max_children)

            if child_list == None or len(child_list) == 0:
                logger.info("mutation to reproduce env FAILED!!!")
                return
            #print(child_list)
            admitted = 0
            for child in child_list:
                new_env_config, new_cppn_params, seed, _, _ = child
                # targeted transfer
                o = self.create_optimizer(new_env_config, new_cppn_params, seed, is_candidate=True)
                score_child, theta_child = o.evaluate_transfer(self.optimizers)
                score_archive, _ = o.evaluate_transfer(self.archived_optimizers, evaluate_proposal=False)
                del o
                if self.pass_mc(score_child):  # check mc
                    self.add_optimizer(env=new_env_config, cppn_params=new_cppn_params, seed=seed, created_at=iteration, model_params=np.array(theta_child))
                    admitted += 1
                    if self.pass_mc(score_archive):
                        self.ANNECS += 1
                    if admitted >= max_admitted:
                        break

            if max_num_envs and len(self.optimizers) > max_num_envs:
                num_removals = len(self.optimizers) - max_num_envs
                self.remove_oldest(num_removals)

    def remove_oldest(self, num_removals):
        list_delete = []
        for optim_id in self.env_registry.keys():
            if len(list_delete) < num_removals:
                list_delete.append(optim_id)
            else:
                break

        for optim_id in list_delete:
            self.archive_optimizer(optim_id)           

    def optimize(self, iterations=200,
                 steps_before_transfer=25,
                 propose_with_adam=False,
                 checkpointing=False,
                 reset_optimizer=True):

        for iteration in range(iterations):

            self.adjust_envs_niches(iteration, self.args.adjust_interval * steps_before_transfer,
                                    max_num_envs=self.args.max_num_envs)

            for o in self.optimizers.values():
                o.clean_dicts_before_iter()

            self.ind_es_step(iteration=iteration)

            if len(self.optimizers) > 1 and iteration % steps_before_transfer == 0:
                self.transfer(propose_with_adam=propose_with_adam,
                              checkpointing=checkpointing,
                              reset_optimizer=reset_optimizer)

            if iteration % steps_before_transfer == 0:
                for o in self.optimizers.values():
                    o.save_to_logger(iteration)
