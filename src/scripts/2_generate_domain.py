import os, sys
ROOT_DIR = os.getcwd()
print(ROOT_DIR)
sys.path.append(f'{ROOT_DIR}/')
import json
import fire
import argparse
from utils.world_generation import WorldGeneration
import random
# random.seed(42)
import multiprocessing
from utils.data_utils import parse_list_by_n

def parse():
    parser = argparse.ArgumentParser()

    # Fill or Path
    parser.add_argument('--api_keys_file', type=str, default='key.txt')
    parser.add_argument('--api_type', type=str, choices=['azure', 'openai', 'mix'], default='openai')
    parser.add_argument('--prompt_file', type=str, default='prompt/desc_evol')
    parser.add_argument('--data_path', type=str)
    # parser.add_argument('--context_path', type=str)
    parser.add_argument('--output_path', type=str) 
    parser.add_argument('--example_num', type=int, default=1)


    # Methodology Config
    parser.add_argument('--max_correction', type=int, default=3)


    # GPT options
    parser.add_argument('--model', type=str, default='gpt-4-0125-preview')
    parser.add_argument('--stop_tokens', type=str, default='\n\n',
                        help='Split stop tokens by ||')

    parser.add_argument('--n_process', type=int)

    # debug options
    parser.add_argument('-v', '--verbose', action='store_false')

    args = parser.parse_args()
    args.stop_tokens = args.stop_tokens.split('||')
    args.prompt_file = f'{ROOT_DIR}/{args.prompt_file}'
    args.output_path = f'{ROOT_DIR}/{args.output_path}'
    args.data_path = f'{ROOT_DIR}/{args.data_path}'
    print("Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args

def main(args):
    data = json.load(open(args.data_path))
    result = []
    for item in data:
        world_generator = WorldGeneration(args)
        success, example = world_generator.close_loop_world_generation(item)
        if 'prompt' in item:
            item.pop('prompt')
        item.update(example)
        if success:
            item['domain'] = item['pred_domain']
            item.pop('pred_domain')
            result.append(item)
    json.dump(obj=result, fp=open(args.output_path, 'w', encoding='utf'), indent=4)


def annotate_single_process(item, args):
    world_generator = WorldGeneration(args)
    success, example = world_generator.close_loop_world_generation(item)
    print(success, example)
    if example is None:
        return False, item
    if 'prompt' in item:
        item.pop('prompt')
    item.update(example)
    item['domain'] = item['pred_domain']
    item.pop('pred_domain')
    return success, item

def multiprocess_main(args):
    data = json.load(open(args.data_path))
    result_all = []
    data_list = parse_list_by_n(data, n=args.n_process)
    for data in data_list:
        pool = multiprocessing.Pool(processes=args.n_process)
        worker_results, result = [], []
        try:
            for item in data:
                worker_results.append(pool.apply_async(annotate_single_process, args=(item, args)))
            for r in worker_results:
                success, item = r.get()
                if success:
                    result.append(item)
        except Exception as e:
            pass
        result_all += result
    json.dump(obj=result_all, fp=open(args.output_path, 'w', encoding='utf'), indent=4)



if __name__ == "__main__":
    # fire.Fire(main)
    args = parse()
    multiprocess_main(args)