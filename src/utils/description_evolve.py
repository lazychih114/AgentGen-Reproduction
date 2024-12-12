import json
import os
from utils.openai_access import Generator
from utils.pddl_utils import extract_pddl, parse_actions, parse_predicates, extract_domain_name
from tqdm import tqdm
import random
import time


class DescEvolve:
    def __init__(self, args) -> None:
        self.prompt_file = args.prompt_file
        self.generator = Generator(args)
        self.model = args.model
        self.args = args

    def _make_evol_prompt(self, context, examples):
        prompt_template = open(self.prompt_file).read()
        prompt = prompt_template.replace('[Context]', context)
        if len(examples) > 1:
            example_text = []
            for idx, example in enumerate(examples):
                example_text.append(f'##Example{idx+1}##\n{example}')
            example_text = example_text.join('\n\n')
        else:
            example_text = examples[0]
        prompt = prompt.replace('[SEED]', example_text)
        return prompt

    def _call_gpt(self, prompt):
        # print(prompt)
        success, gpt_response, tokens = self.generator.generate(prompt, model=self.model)
        # print(gpt_response)
        if success is True:
            gpt_response = gpt_response[0]
            print(gpt_response)
            # domain_pddl = extract_pddl(gpt_response)
            return gpt_response, tokens['total']
        else:
            return None, None

    def _call_gpt_multiprocess(self, prompt_list):
        result_list = self.generator.generate_multiprocess(prompt_list, model=self.model)
        parsed_result = []
        for success, gpt_response, tokens in result_list:
            if success is True:
                gpt_response = gpt_response[0]
                print(gpt_response)
                # domain_pddl = extract_pddl(gpt_response)
                parsed_result.append((gpt_response, tokens['total']))
            else:
                parsed_result.append((None, None))
        return parsed_result

    def close_loop_evol(self, context, examples):
        '''
        data: data is a dictionary contains "description" (mandatory), "unfilled_domain", "example_trajectory" (optional, only under "fillin" setting), 
        '''
        st = time.time()
        prompt = self._make_evol_prompt(context, examples)
        description, total_token = self._call_gpt(prompt)
        
        if not self.args.verbose:
            return {'description': description, 'time': time.time() - st, 'token': total_token}
        else:
            return {'description': description, 'time': time.time() - st, 'token': total_token, 'prompt': prompt}

    def close_loop_evol_multiprocess(self, context_list, examples_list):
        '''
        data: data is a dictionary contains "description" (mandatory), "unfilled_domain", "example_trajectory" (optional, only under "fillin" setting), 
        '''
        st = time.time()
        prompt_list = [self._make_evol_prompt(context, examples) for context, examples in zip(context_list, examples_list)]
        result_list = self._call_gpt_multiprocess(prompt_list)
        items = []
        for prompt, (description, total_token) in zip(prompt_list, result_list):
            if not self.args.verbose:
                item = {'description': description, 'time': time.time() - st, 'token': total_token}
            else:
                item = {'description': description, 'time': time.time() - st, 'token': total_token, 'prompt': prompt}
            items.append(item)
        return items