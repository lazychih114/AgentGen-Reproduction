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
from utils.pddl_utils import extract_pddl, extract_domain_name
import copy
from utils.pddl_utils import parse_actions, parse_predicates
from utils.data_utils import extract_from_python, parse_list_by_n
import multiprocessing
import re

def parse():
    parser = argparse.ArgumentParser()

    # Fill or Path
    parser.add_argument('--api_keys_file', type=str, default='key.txt')
    parser.add_argument('--api_type', type=str, choices=['azure', 'openai', 'mix'], default='azure')
    parser.add_argument('--prompt_dir', type=str, default='prompt')
    parser.add_argument('--data_path', type=str)
    # parser.add_argument('--context_path', type=str)
    parser.add_argument('--output_path', type=str) 


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
    args.prompt_dir = f'{ROOT_DIR}/{args.prompt_dir}'
    args.output_path = f'{ROOT_DIR}/{args.output_path}'
    args.data_path = f'{ROOT_DIR}/{args.data_path}'
    print("Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args

def count_arg_patterns(dictionary):
    counts = {}
    for key, value in dictionary.items():
        pattern = r"\{arg\d+\}"
        matches = re.findall(pattern, value)
        counts[key] = len(matches)
    return counts


def nl_interface_check(domain, nl_interface):
    predicate_map, action_map = parse_predicates(domain), parse_actions(domain)
    predicate_map.update(action_map)
    raw_nli = copy.deepcopy(nl_interface)
    # nl_interface = {k:v.count('{}') for k,v in nl_interface.items()}
    nl_interface = count_arg_patterns(nl_interface)
    nl_interface = {_.lower():v for _, v in nl_interface.items()}
    raw_nli = {_.lower():v for _, v in raw_nli.items()}
    if not len(predicate_map) == len(nl_interface):
        if len(predicate_map) < len(nl_interface):
            redundant_keys = list(set(nl_interface.keys()) - set(predicate_map.keys()))
            redundant_keys = ','.join(redundant_keys)
            return False, f'The number of keys in natural language interface is not equal to the number of actions and predicates in the pddl. Redundant Keys: {redundant_keys}'
        else:
            missing_keys = list(set(predicate_map.keys()) - set(nl_interface.keys()))
            missing_keys = ','.join(missing_keys)
            return False, f'The number of keys in natural language interface is not equal to the number of actions and predicates in the pddl. Missing Keys: {missing_keys}'
    for k, v in predicate_map.items():
        if not k in nl_interface:
            return False, f'Key "{k}" is not in natural language interface'
        elif not v == nl_interface[k]:
            sent = raw_nli[k]
            na = nl_interface[k]
            return False, f'The arity of "{k}" should be {v}, but the arity of "{k}" in natural language interface "{sent}" is {na}'
    return True, ''

def close_loop_nl_interface_generation(args, generator:Generator, prompt_template, domain, description):
    prompt = prompt_template.replace('[PDDL_Domain]', domain).replace('[PDDL_Description]', description)
    trace = []
    max_retry, cnt_retry = 3, 0
    domain_name = extract_domain_name(domain)
    # model = 'gpt-4-0125-preview'
    while cnt_retry < max_retry:
        success, gpt_response, tokens = generator.generate(prompt, model=args.model)
        if success:
            gpt_response = gpt_response[0]
        print(gpt_response)
        try:
            nl_interface_str = extract_from_python(gpt_response)
            success, error_info = nl_interface_check(domain, eval(nl_interface_str))
        except Exception as e:
            success, error_info = False, 'Error when parsing. There exists syntax errors.'
        trace.append({'nl': nl_interface_str, 'gpt_response': gpt_response})
        if success:
            print(f'{domain_name} successfully generate nl interface')
            return eval(nl_interface_str.strip()), trace
        else:
            print(f'{domain_name} failed when generating nl interface. error info: {error_info}')
            cnt_retry += 1
            prompt += f'\nThe generated natural language interface {nl_interface_str} occurs error: {error_info}. You need to carefully review your answer and output the correct natural language interface. The corrected natural language interface should also be wrapped in ```python```:\n'
            trace[-1].update({'error_info': error_info, 'prompt': prompt})
    return None, trace

def open_loop_nl_interface_generation(args, generator:Generator, prompt_template, domain, description):
    prompt = prompt_template.replace('[PDDL_Domain]', domain).replace('[PDDL_Description]', description)
    domain_name = extract_domain_name(domain)
    # model = 'gpt-4-0125-preview'
    success, gpt_response, tokens = generator.generate(prompt, model=args.model)
    if success:
        gpt_response = gpt_response[0]
    print(gpt_response)
    nl_interface_str = extract_from_python(gpt_response)
    return eval(nl_interface_str.strip()), []

def main(args):
    data = json.load(open(args.data_path))
    result = []
    generator = Generator(args)
    prompt_template = open(f'{args.prompt_dir}/nl_interface_generation_trimmed').read()
    for idx, item in enumerate(tqdm(data)):
        domain, description = item['domain'], item['description']
        nl_interface, trace = close_loop_nl_interface_generation(args, generator, prompt_template, domain, description)
        # if nl_interface is not None:
        item.update({'nl_interface': nl_interface, 'nl_interface_debug': trace})
        result.append(item)
    json.dump(obj=result, fp=open(args.output_path, 'w', encoding='utf'), indent=4)

def annotate_single_process(args, prompt_template, item, pid):
    os.environ['PID'] = str(pid)
    generator = Generator(args)
    domain, description = item['domain'], item['description']
    try:
        nl_interface, trace = close_loop_nl_interface_generation(args, generator, prompt_template, domain, description)
    except Exception as e:
        nl_interface, trace = {}, []
    # if nl_interface is not None:
    item.update({'nl_interface': nl_interface, 'nl_interface_debug': trace})
    return item

def multiprocess_main(args):
    data = json.load(open(args.data_path))
    result_all = []
    data_list = parse_list_by_n(data, n=args.n_process)
    prompt_template = open(f'{args.prompt_dir}/nl_interface_generation_trimmed').read()
    for data in data_list:
        pool = multiprocessing.Pool(processes=args.n_process)
        worker_results, result = [], []
        for pid, item in enumerate(data):
            worker_results.append(pool.apply_async(annotate_single_process, args=(args, prompt_template, item, pid)))
        for r in worker_results:
            item = r.get()
            # if item['nl_interface'] is not None:
            if item['nl_interface'] is None:
                item['nl_interface'] = {}
            result.append(item)
        result_all += result
    json.dump(obj=result_all, fp=open(args.output_path, 'w', encoding='utf'), indent=4)

if __name__ == "__main__":
    # fire.Fire(main)
    args = parse()
    # main(args)
    multiprocess_main(args)