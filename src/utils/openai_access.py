import openai
import time
import requests
from openai import OpenAI
import multiprocessing


def call_chatgpt_openai(ins, keys, model="gpt-3.5-turbo", n=1, temperature=0):
    key_index = 0
    def get_openai_completion(prompt, api_key):
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model= model,
            n= n,
            messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                
                ],
            temperature=temperature,
            max_tokens=2048,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
            )
        res = [_.message.content for _ in response.choices]
        tokens = response.usage
        tokens = {'prompt': tokens.prompt_tokens, 'completion': tokens.completion_tokens, 'total': tokens.total_tokens}
        return res, tokens
    success = False
    re_try_count = 10
    ans = ''
    while not success and re_try_count >= 0:
        re_try_count -= 1
        key = keys[key_index]
        try:
            ans, tokens = get_openai_completion(ins, key)
            success = True
        except Exception as e:
            key_index  = (key_index + 1) % len(keys)
            print(str(e))
            time.sleep(5)
            print('retry for sample:', ins)
    return success, ans, tokens


def call_chatgpt_azure(ins, model="gpt-3.5-turbo", n=1, temperature=0):
    pass



class Generator:
    def __init__(self, args) -> None:
        self.api_type = args.api_type
        if self.api_type in ['mix', 'openai']:
            self.keys = [line.strip() for line in open(args.api_keys_file).readlines()]


    def generate(self, prompt, model="gpt-3.5-turbo", n=1, temperature=0):
        api_type = self.api_type
        if api_type == 'openai':
            return call_chatgpt_openai(prompt, self.keys, model, n, temperature)
        elif api_type == 'azure':
            return call_chatgpt_azure(prompt, model, n, temperature)
        elif api_type == 'mix':
            success, gpt_response, tokens = call_chatgpt_azure(prompt, model=model, temperature=temperature, n=n)
            if not success or gpt_response[0] is None: # failed
                return call_chatgpt_openai(prompt, keys=self.keys, model=model, n=n, temperature=temperature)
            else:
                return True, gpt_response, tokens
        else:
            raise Exception(f'{api_type} Not Implemented')
    
    def generate_multiprocess(self, prompt_list, model="gpt-3.5-turbo", n=1, temperature=0):
        print('Generate with multiprocessing, number of processes:', len(prompt_list))
        pool = multiprocessing.Pool(processes=len(prompt_list))
        worker_results = []
        result = []
        for prompt in prompt_list:
            worker_results.append(pool.apply_async(self.generate, args=(prompt, model, n, temperature)))
        for r in worker_results:
            result.append(r.get())
        return result
