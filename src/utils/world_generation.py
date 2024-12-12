import json
import os
from utils.openai_access import Generator
from utils.pddl_utils import extract_pddl, parse_actions, parse_predicates, extract_domain_name
from tqdm import tqdm
import random
import time


correction_format = '''
Round [Round]
Incorrect PDDL:
[PDDL]
Error Information:
[Error]
Corrected PDDL:
[Corrected_PDDL]
'''

history_format = '''
Round [Round]
Incorrect PDDL: 
[PDDL]
Error Information:
[Error]
'''

def make_correction_prompt(prompt, trace, domain, error_info):
    # .replace('[Corrected_PDDL]', _['corrected_domain'])
    history = [history_format.replace('[Round]', str(idx)).replace('[PDDL]', _['incorrect_domain']).replace('[Error]', _['error_info']) for idx, _ in enumerate(trace)]
    history.append(correction_format.replace('[Round]', str(len(trace))).replace('[PDDL]', domain).replace('[Corrected_PDDL]', '').replace('[Error]', error_info))
    prompt += '\n\n' + '\n\n'.join(history)
    return prompt


from pddlgym.parser import PDDLDomainParser
import copy, traceback
def _checker(_domain):
    temp_path = './temp.pddl'
    open(temp_path, 'w', encoding='utf').write(_domain)
    try:
        PDDLDomainParser(domain_fname=temp_path, operators_as_actions=True, expect_action_preds=False)
        return True, 'Success'
    except Exception as e:
        exception_type = type(e).__name__
        traceback_info = traceback.format_exc()
        error_message = f"{exception_type}: {str(e)}\nTraceback:\n{traceback_info}"
        return False, error_message


domain_correct_prompt = '''
I would like you to serve as an expert in PDDL, assisting me in correcting erroneous PDDL code. I will provide you with the incorrect PDDL along with the error messages returned by the system. You should output your thought process firstly. You MUST enclose the COMPLETE corrected PDDL within ```pddl```.
Here are some hints to help you debug the pddl domain file:
1. You should start by checking if all the essential domain constructs or domain definition constructs are present. Commonly included domains comprise:
    a. :domain declaration to name the domain.
    b. :requirements to specify the PDDL features used in the domain.
    c. :types to define any object types for categorizing entities in the planning problem.
    d. :constants (if necessary) to declare any objects that remain static throughout the planning problems.
    e. :predicates to define the properties and relations between objects that can change over time.
    f. :functions (if necessary) to define numeric functions for more complex domains.
    g. :action definitions for each action that agents can perform, including parameters, preconditions, and effects.
2. You need to check the number of parameters of each actions.
3. Having :typing in the domain indicates that it uses a hierarchy to organize objects. Therefore, it's crucial to clearly list all object types related to the planning task in a :types section.
4. '-' should not appear in :types.
'''


class WorldGeneration:
    def __init__(self, args) -> None:
        self.prompt_file = args.prompt_file
        self.max_correction = args.max_correction
        self.generator = Generator(args)
        self.model = args.model

    def _make_domain_generation_prompt(self, data):
        prompt_template = open(self.prompt_file).read()
        description = data['description']
        prompt = prompt_template.replace('[Description]', description)
        return prompt


    def _domain_generation(self, prompt):
        # print(prompt)
        success, gpt_response, tokens = self.generator.generate(prompt, model=self.model)
        # print(gpt_response)
        gpt_response = gpt_response[0]
        print(gpt_response)
        domain_pddl = extract_pddl(gpt_response)
        return domain_pddl, tokens['total']


    def _domain_correction(self, domain, max_retry=3):
        domain_name = extract_domain_name(domain)
        count = 0
        trace = []
        token = 0
        while True:
            success, error_info = _checker(domain)
            if success:
                print(f'Env {domain_name} Passed Test')
                return success, domain, trace, token
            elif count == max_retry:
                print(f'Env {domain_name} Retry Exceeded')
                return False, None, trace, token
            else:
                print(f'Env {domain_name}, Correct Round {count}. Error Info: {error_info}')
                count += 1
                # prompt = open(f'{prompt_dir}/domain_correct').read()
                prompt = domain_correct_prompt
                # prompt = prompt.replace('[PDDL]', domain).replace('[Error]', error_info)
                prompt = make_correction_prompt(prompt, trace, domain, error_info)
                # model='gpt-4-0125-preview'
                # text = call_chatgpt(prompt, temperature=0)[0]
                success, text, tokens = self.generator.generate(prompt, temperature=0, model=self.model)
                text = text[0]
                token += tokens['total']
                pre_domain = domain
                print(domain)
                domain = extract_pddl(text)
                trace.append({'error_info': error_info, 'incorrect_domain': pre_domain, 'corrected_domain': domain, 'gpt_response': text, 'prompt': prompt})
        # return corrected_env, corrected_env['domain_pddl'] is not None


    def close_loop_world_generation(self, data):
        '''
        data: data is a dictionary contains "description" (mandatory), "unfilled_domain", "example_trajectory" (optional, only under "fillin" setting), 
        '''
        st = time.time()
        prompt = self._make_domain_generation_prompt(data)
        try:
            domain, total_token = self._domain_generation(prompt)
            print(domain)
            success, corrected_env, trace, token = self._domain_correction(domain, self.max_correction)
            total_token += token
            return success, {'pred_domain': corrected_env, 'correct_trace':trace, 'time': time.time() - st, 'token': total_token}
        except Exception as e:
            return False, None

        