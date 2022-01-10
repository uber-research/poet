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
from poet_distributed.reproduce_ops import Reproducer
from poet_distributed.novelty import compute_novelty_vs_archive
import json


def construct_niche_fns_from_env(args, env, seed):
    def niche_wrapper(configs, seed):  # force python to make a new lexical scope
        def make_niche():
            from poet_distributed.niches import Box2DNiche
            return Box2DNiche(env_configs=configs,
                            seed=seed,
                            init=args.init,
                            stochastic=args.stochastic)

        return make_niche

    niche_name = env.name
    configs = (env,)

    return niche_name, niche_wrapper(list(configs), seed)


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

        self.env_registry = OrderedDict()
        self.env_archive = OrderedDict()
        self.env_reproducer = Reproducer(args)
        self.optimizers = OrderedDict()

        if args.start_from:
            logger.debug("args.start_from {}".format(args.start_from))
            with open(args.start_from) as f:
                start_from_config = json.load(f)

            logger.debug(start_from_config['path'])
            logger.debug(start_from_config['niches'])
            logger.debug(start_from_config['exp_name'])

            path = start_from_config['path']
            exp_name = start_from_config['exp_name']
            prefix = path + exp_name +'/'+exp_name+'.'
            for niche_name, niche_file in sorted(start_from_config['niches'].items()):
                logger.debug(niche_name)
                niche_file_complete = prefix + niche_file
                logger.debug(niche_file_complete)
                with open(niche_file_complete) as f:
                    data = json.load(f)
                    logger.debug('loading file %s' % (niche_file_complete))
                    model_params = np.array(data[0])  # assuming other stuff is in data
                    logger.debug(model_params)

                env_def_file = prefix + niche_name + '.env.json'
                with open(env_def_file, 'r') as f:
                    exp = json.loads(f.read())

                env = Env_config(**exp['config'])
                logger.debug(env)
                seed = exp['seed']
                self.add_optimizer(env=env, seed=seed, model_params=model_params)

        else:
            env = Env_config(
                name='flat',
                ground_roughness=0,
                pit_gap=[])

            self.add_optimizer(env=env, seed=args.master_seed)

    def create_optimizer(self, env, seed, created_at=0, model_params=None, is_candidate=False):

        assert env != None

        optim_id, niche_fn = construct_niche_fns_from_env(args=self.args, env=env, seed=seed)

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


    def add_optimizer(self, env, seed, created_at=0, model_params=None):
        '''
            creat a new optimizer/niche
            created_at: the iteration when this niche is created
        '''
        o = self.create_optimizer(env, seed, created_at, model_params)
        optim_id = o.optim_id
        self.optimizers[optim_id] = o

        assert optim_id not in self.env_registry.keys()
        assert optim_id not in self.env_archive.keys()
        self.env_registry[optim_id] = env
        self.env_archive[optim_id] = env
        #dump the env
        log_file = self.args.log_file
        env_config_file = log_file + '/' + log_file.split('/')[-1] + '.' + optim_id + '.env.json'
        record = {'config': env._asdict(), 'seed': seed}
        with open(env_config_file,'w') as f:
            json.dump(record, f)

    def delete_optimizer(self, optim_id):
        assert optim_id in self.optimizers.keys()
        #assume optim_id == env_id for single_env niches
        o = self.optimizers.pop(optim_id)
        del o
        assert optim_id in self.env_registry.keys()
        self.env_registry.pop(optim_id)
        logger.info('DELETED {} '.format(optim_id))

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
        for source_optim in self.optimizers.values():
            source_tasks = []
            for target_optim in [o for o in self.optimizers.values()
                                    if o is not source_optim]:
                task = target_optim.start_theta_eval(
                    source_optim.theta)
                source_tasks.append((task, target_optim))

            for task, target_optim in source_tasks:
                stats = target_optim.get_theta_eval(task)

                target_optim.update_dicts_after_transfer(source_optim_id=source_optim.optim_id,
                    source_optim_theta=source_optim.theta,
                    stats=stats, keyword='theta')

        logger.info('Computing proposal transfers...')
        for source_optim in self.optimizers.values():
            source_tasks = []
            for target_optim in [o for o in self.optimizers.values()
                                    if o is not source_optim]:
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
        parent = self.env_registry[optim_id]
        child_env_config = self.env_reproducer.mutate(parent)

        logger.info("we pick to mutate: {} and we got {} back".format(optim_id, child_env_config.name))
        logger.debug("parent")
        logger.debug(parent)
        logger.debug("child")
        logger.debug(child_env_config)

        seed = np.random.randint(1000000)
        return child_env_config, seed, optim_id

    def get_child_list(self, parent_list, max_children):
        child_list = []

        mutation_trial = 0
        while mutation_trial < max_children:
            new_env_config, seed, parent_optim_id = self.get_new_env(parent_list)
            mutation_trial += 1
            if self.pass_dedup(new_env_config):
                o = self.create_optimizer(new_env_config, seed, is_candidate=True)
                score = o.evaluate_theta(self.optimizers[parent_optim_id].theta)
                del o
                if self.pass_mc(score):
                    novelty_score = compute_novelty_vs_archive(self.env_archive, new_env_config, k=5)
                    logger.debug("{} passed mc, novelty score {}".format(score, novelty_score))
                    child_list.append((new_env_config, seed, parent_optim_id, novelty_score))

        #sort child list according to novelty for high to low
        child_list = sorted(child_list,key=lambda x: x[3], reverse=True)
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

            child_list = self.get_child_list(list_repro, max_children)

            if child_list == None or len(child_list) == 0:
                logger.info("mutation to reproduce env FAILED!!!")
                return
            #print(child_list)
            admitted = 0
            for child in child_list:
                new_env_config, seed, _, _ = child
                # targeted transfer
                o = self.create_optimizer(new_env_config, seed, is_candidate=True)
                score_child, theta_child = o.evaluate_transfer(self.optimizers)
                del o
                if self.pass_mc(score_child):  # check mc
                    self.add_optimizer(env=new_env_config, seed=seed, created_at=iteration, model_params=np.array(theta_child))
                    admitted += 1
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
            self.delete_optimizer(optim_id)

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
