# from . import tests
# from . import core
# from . import structs
# from . import spaces

import matplotlib
# matplotlib.use("Agg")
from pddlgym.rendering import *
from gym.envs.registration import register
from pddlgym_planners.fd import FD
from pddlgym.parser import PDDLDomainParser, PDDLProblemParser
import pddlgym
from utils.env_utils import get_goal_and_obs, literal_to_text, get_obs_text, get_action_space
import gym
from tqdm import tqdm
from utils.pddl_utils import extract_domain_name

import os
import re

# Save users from having to separately import gym
def make(*args, **kwargs):
    # env checker fails since obs is not an numpy array like object
    return gym.make(*args, disable_env_checker=True, **kwargs)


def custom_register_pddl_env(domain_pddl, problem_pddls, is_test_env=False, other_args={'operators_as_actions' : True, 'dynamic_action_space' : True, "raise_error_on_invalid_action": False}):
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "temp_pddl")
    os.makedirs(dir_path, exist_ok=True)
    name = extract_domain_name(domain_pddl)
    # FIXME temp process
    if name in ['baking', 'doors']:
        other_args['operators_as_actions'] = False
    else:
        other_args['operators_as_actions'] = True
    print(f'Constructing Temp Environment for {name}...')
    domain_file = os.path.join(dir_path, "{}.pddl".format(name.lower()))
    with open(domain_file, 'w', encoding='utf') as f:
        f.write(domain_pddl)
        f.close()
    gym_name = name.capitalize()
    problem_dirname = name.lower()
    problem_dir = os.path.join(dir_path, problem_dirname)
    if os.path.exists(problem_dir):
        shutil.rmtree(problem_dir)
    os.makedirs(problem_dir)
    # os.makedirs(problem_dir, exist_ok=True) # 删除掉之前的
    for idx, pp in enumerate(problem_pddls):
        with open(f'{problem_dir}/problem{idx}.pddl', 'w', encoding='utf') as f:
            f.write(pp)
            f.close()
    env_id = 'PDDLEnv{}-v0'.format(gym_name)
    register(
        id=env_id,
        entry_point='pddlgym.core:PDDLEnv',
        kwargs=dict({'domain_file' : domain_file, 'problem_dir' : problem_dir,
                     **other_args}),
    )
    print(open(domain_file).read())
    return env_id

from pddlgym.parser import PDDLDomainParser, PDDLProblemParser
import glob
def load_pddl(domain_file, problem_dir, operators_as_actions=False):
    """
    Parse domain and problem PDDL files.

    Parameters
    ----------
    domain_file : str
        Path to a PDDL domain file.
    problem_dir : str
        Path to a directory of PDDL problem files.
    operators_as_actions : bool
        See class docstirng.

    Returns
    -------
    domain : PDDLDomainParser
    problems : [ PDDLProblemParser ]
    """
    domain = PDDLDomainParser(domain_file, 
        expect_action_preds=(not operators_as_actions),
        operators_as_actions=operators_as_actions)
    problems = []
    problem_files = [f for f in glob.glob(os.path.join(problem_dir, "*.pddl"))]
    for problem_file in sorted(problem_files):
        problem = PDDLProblemParser(problem_file, domain.domain_name, 
            domain.types, domain.predicates, domain.actions, domain.constants)
        problems.append(problem)
    return domain, problems

import shutil

def _check_ppddl(domain_pddl, problem_pddl):
    try:
        pddlgym.make(custom_register_pddl_env(domain_pddl, problem_pddls=[problem_pddl]))
        return True
    except Exception as e:
        return False

def check_problem_pddls(domain_pddl, problem_pddls):    # syntax check
    new_pp = []
    for problem in problem_pddls:
        if _check_ppddl(domain_pddl, problem):
            new_pp.append(problem)
    return new_pp


def gen_traj_from_pddl(env, problem_index, planning_algo, predicate_map=None):
    env.fix_problem_index(problem_index)
    obs, debug_info = env.reset()
    cur_example = {}
    obs, debug_info = env.reset()
    goal_text, state_text, _ = get_goal_and_obs(obs, predicate_map)
    cur_example.update({'goal':goal_text, 'trajectory':[{'observation':state_text}]})
    # seq-opt-lmcut
    planner = FD(alias_flag=f"--alias {planning_algo}")
    plan = planner(env.domain, obs)
    for act in plan:
        act_text = literal_to_text(act, predicate_map)
        obs, reward, done, truncated, debug_info = env.step(act)
        # 是否记录原来的action和observation
        cur_example['trajectory'].append({'action': act_text, 'observation': get_obs_text(obs, predicate_map=predicate_map), 'reward': reward})
    # json.dump(obj=cur_example, fp=open("./custom_example", 'w', encoding='utf'), indent=4)
    return cur_example

import traceback

def gen_traj_batch(domain_pddl, problem_pddls, planning_algos, predicate_map=None):
    data = []
    for problem_index, problem in enumerate(problem_pddls):
        plan_set = set()
        try:
            pddl_env = pddlgym.make(custom_register_pddl_env(domain_pddl, [problem]))
        except Exception as e:
            print(f'{problem_index} {str(e)}, Trace: {traceback.format_exc()}')
            continue
        for traj_id, algo in enumerate(planning_algos):
            example = {'problem_id': problem_index, 'planning_algorithm': algo}
            try:
                example.update(gen_traj_from_pddl(pddl_env, 0, algo, predicate_map))
                plan = '\n'.join([_['action'] for _ in example['trajectory'][1:]])
                if plan not in plan_set and len(example['trajectory']) > 1:
                    plan_set.add(plan)
                    data.append(example)
                print(f'{problem_index} success')
            except Exception as e:
                print(f'{problem_index} {str(e)}, Trace: {traceback.format_exc()}')
                if "AttributeError: 'NoneType' object has no attribute 'start'" in traceback.format_exc():
                    print(domain_pddl)
            finally:
                # bar.update(1)
                pass
    print(f'Total Number of Trajectories: {len(data)}')
    return data