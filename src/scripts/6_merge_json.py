import os, sys
ROOT_DIR = os.getcwd()
print(ROOT_DIR)
sys.path.append(f'{ROOT_DIR}/')
import json
import fire
from utils.pddlgym_utils import gen_traj_batch
from utils.data_utils import traj2gpt_wonl


def main(d1_path, d2_path, output_path):
    d1 = json.load(open(d1_path))
    d2 = json.load(open(d2_path))
    json.dump(obj=d1+d2, fp=open(output_path, 'w', encoding='utf'), indent=4)
    print(f'Path {d1_path} & {d2_path}\n{len(d1), len(d2), len(d1+d2)}')


if __name__ == "__main__":
    fire.Fire(main)