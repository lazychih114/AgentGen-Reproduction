import os, sys
ROOT_DIR = os.getcwd()
print(ROOT_DIR)
sys.path.append(f'{ROOT_DIR}/')
import json
import fire
import argparse
import random
# random.seed(42)
from utils.openai_access import Generator
from tqdm import tqdm
from utils.pddl_utils import extract_pddl
from utils.data_utils import parse_list_by_n
import multiprocessing

def parse():
    parser = argparse.ArgumentParser()

    # Fill or Path
    parser.add_argument('--api_keys_file', type=str, default='key.txt')
    parser.add_argument('--api_type', type=str, choices=['azure', 'openai', 'mix'], default='azure')
    parser.add_argument('--prompt_dir', type=str, default='prompt')
    parser.add_argument('--data_path', type=str)
    # parser.add_argument('--context_path', type=str)
    parser.add_argument('--output_path', type=str) 
    parser.add_argument('--example_num', type=int, default=1)


    # Methodology Config
    parser.add_argument('--max_correction', type=int, default=3)
    parser.add_argument("--prob_num", type=int, default=5)
    parser.add_argument('--n_process', type=int)

    # GPT options
    parser.add_argument('--model', type=str, default='gpt-4-0125-preview')
    parser.add_argument('--stop_tokens', type=str, default='\n\n',
                        help='Split stop tokens by ||')
    

    # debug options
    parser.add_argument('-v', '--verbose', action='store_false')

    args = parser.parse_args()
    args.stop_tokens = args.stop_tokens.split('||')
    args.prompt_dir = f'{ROOT_DIR}/{args.prompt_dir}'
    args.output_path = f'{ROOT_DIR}/{args.output_path}'
    args.data_path = f'{ROOT_DIR}/{args.data_path}'
    print("Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def domain2problem_zero_shot(args, domain, generator:Generator, prob_num):
    prompt = open(f'{args.prompt_dir}/problem_generation_zero_shot').read()
    prompt = prompt.replace('[Domain]', domain)
    success, result, _ = generator.generate(prompt, model=args.model, temperature=0.5, n=prob_num)
    result = list(set([_.strip() for _ in result]))
    result = list(set([extract_pddl(_) for _ in result]))
    return result

def domain2problem_evol(args, domain, seed_problem, generator:Generator):
    methods = [
        # Harder
        'Increase the number of objects',
        'Modify the goal conditions to make it harder',
        # Easier
        'Decrease the number of objects to make it easier',
        'Modify the goal conditions to make it easier',
        # Others
        'Modify the initial state or properties of objects',
        'Change the initial placement or locations of objects',
        'Change the constraints in the problem file',
        'Adjust the object requirements in the initial state',
        'Modify the allowed object combinations or configurations'
    ]
    prompt = open(f'{args.prompt_dir}/problem_generation_evol').read()
    prompt = prompt.replace('[Domain]', domain).replace('[Example_Problem]', seed_problem).replace('[Method]', random.sample(methods, k=1)[0])
    success, result, _ = generator.generate(prompt, model=args.model, temperature=0.5)
    print(result)
    result = result[0].strip()
    result = extract_pddl(result)
    return result




def main(args):
    data = json.load(open(args.data_path))
    result = []
    generator = Generator(args)
    for item in tqdm(data):
        domain = item['pred_domain']
        if domain is not None:
            seed_probs = domain2problem_zero_shot(args, domain, generator, args.prob_num)
            evol_probs = []
            for _ in range(args.prob_num):
                evol_probs.append(domain2problem_evol(args, domain ,random.sample(seed_probs, k=1)[0], generator))
            evol_probs = list(set(evol_probs))
            item.update({'domain': item['pred_domain'], 'problems': seed_probs + evol_probs})
            result.append(item)
    json.dump(obj=result, fp=open(args.output_path, 'w', encoding='utf'), indent=4)

def annotate_single_process(args, item):
    generator = Generator(args)
    # print(set(item.keys()))
    domain, description = item['domain'], item['description']
    seed_probs = domain2problem_zero_shot(args, domain, generator, args.prob_num)
    evol_probs = []
    for _ in range(args.prob_num):
        evol_probs.append(domain2problem_evol(args, domain ,random.sample(seed_probs, k=1)[0], generator))
    evol_probs = list(set(evol_probs))
    item.update({'problems': seed_probs + evol_probs})
    return item

def multiprocess_main(args):
    data = json.load(open(args.data_path))
    result_all = []
    data_list = parse_list_by_n(data, n=args.n_process)
    for data in tqdm(data_list):
        pool = multiprocessing.Pool(processes=args.n_process)
        worker_results, result = [], []
        for item in data:
            worker_results.append(pool.apply_async(annotate_single_process, args=(args, item)))
        for r in worker_results:
            item = r.get()
            result.append(item)
        result_all += result
        # break
    json.dump(obj=result_all, fp=open(args.output_path, 'w', encoding='utf'), indent=4)


if __name__ == "__main__":
    # fire.Fire(main)
    args = parse()
    multiprocess_main(args)