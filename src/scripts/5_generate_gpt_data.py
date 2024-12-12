import os, sys
ROOT_DIR = os.getcwd()
print(ROOT_DIR)
sys.path.append(f'{ROOT_DIR}/')
import json
import fire
from utils.pddlgym_utils import gen_traj_batch
from utils.data_utils import traj2gpt_wonl, traj2gpt_wonl_open_loop


def main(data_path, output_path, solvable_path=None, close_loop=True):
    algo_list = ["seq-opt-lmcut"]
    data = json.load(open(data_path))
    solvable_envs = []
    gpt_list = []
    for item in data:
        domain, problems = item['domain'], item['problems']
        traj_data = gen_traj_batch(domain, problems, algo_list, predicate_map=item['nl_interface'])
        if len(traj_data) > 0 and solvable_path is not None:
            solvable_envs.append(item)
        if close_loop:
            for idx, traj in enumerate(traj_data):
                gpt_data = traj2gpt_wonl(item, traj, idx)
                gpt_list.append(gpt_data)
        else:
            for idx, traj in enumerate(traj_data):
                gpt_data = traj2gpt_wonl_open_loop(item, traj, idx)
                gpt_list.append(gpt_data)

    json.dump(obj=gpt_list, fp=open(output_path, 'w', encoding='utf'), indent=4)
    if solvable_path is not None:
        json.dump(obj=solvable_envs, fp=open(solvable_path, 'w', encoding='utf'), indent=4)


if __name__ == "__main__":
    fire.Fire(main)