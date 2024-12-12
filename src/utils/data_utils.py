import json
import os
from utils.pddl_utils import extract_actions, extract_domain_name

# env -> keys: description, domain, domain_name, trajectory
def traj2gpt_wonl(env, traj, idx, sliding_window=False):
    example = {}
    description = env['description']
    domain_pddl, domain_name = env['domain'], extract_domain_name(env['domain'])
    action_list = extract_actions(domain_pddl)
    # action_interface = {act:nl_interface[act] for act in action_list}
    system_prompt = 'You are a master in planning. Generate your next step of action after Action. The output format should be: Action: {corresponding_action}'
    # action_interface = '\n'.join(list(action_interface.values()))
    # addtion_desc = addtion_description.replace('[NL_Interface]', action_interface)
    # instruction = f'{description}\n{addtion_desc}'
    instruction = description
    goal, trajectory = traj['goal'], traj['trajectory']
    init_obs = trajectory[0]['observation']
    instruction += f'\nGoal: {goal}\nObservation:{init_obs}\n'
    conversations = []
    conversations.append({'from': 'human', 'value':instruction})
    trajectory = trajectory[1:]
    for _i, d in enumerate(trajectory):
        act, obs = d['action'], d['observation']
        conversations.append({'from': 'gpt', 'value': f'Action: {act}\n'})
        if _i != len(trajectory) - 1:
            conversations.append({'from': 'human', 'value': f'Observation: {obs}\n'})
    example.update({'task': 'pddl', 'id': f'{domain_name}_{idx}', 'system_prompt': system_prompt, 'conversations': conversations})
    return example

def traj2gpt_wonl_open_loop(env, traj, idx, sliding_window=False):
    example = {}
    description = env['description']
    domain_pddl, domain_name = env['domain'], extract_domain_name(env['domain'])
    action_list = extract_actions(domain_pddl)
    # action_interface = {act:nl_interface[act] for act in action_list}
    system_prompt = 'You are a master in open-loop planning. Generate your plan (action list). The output format should be: Plan:\n {corresponding_action_list}'
    # action_interface = '\n'.join(list(action_interface.values()))
    # addtion_desc = addtion_description.replace('[NL_Interface]', action_interface)
    # instruction = f'{description}\n{addtion_desc}'
    instruction = description
    goal, trajectory = traj['goal'], traj['trajectory']
    init_obs = trajectory[0]['observation']
    instruction += f'\nGoal: {goal}\nObservation:{init_obs}\n'
    conversations = []
    conversations.append({'from': 'human', 'value':instruction})
    trajectory = trajectory[1:]
    plan = [d['action'] for _i, d in enumerate(trajectory)]
    plan = '\n'.join(plan)
    conversations.append({'from': 'gpt', 'value':f'Plan:\n {plan}'})
    example.update({'task': 'pddl', 'id': f'{domain_name}_{idx}', 'system_prompt': system_prompt, 'conversations': conversations})
    return example


import re
def extract_from_python(text):
   pattern = r"```python\n(.*?)```"
   # 使用re.DOTALL使得点号(.)可以匹配包括换行符在内的任意字符
   matches = re.findall(pattern, text, re.DOTALL)
   # print(text)
   # if matches:
   # 返回第一个匹配的PDDL文本块，假设文档中只有一个PDDL代码块
   p = matches[0].replace('```python', '').replace('```', '').strip()
   return p



def parse_list_by_n(lst, n):
    """
    将列表lst拆分为n个大小相等的子列表
    
    Args:
        lst (list): 需要被拆分的列表
        n (int): 每个子列表的元素个数
        
    Returns:
        list: 包含n个子列表的列表
    """
    result = []
    length = len(lst)
    for i in range(0, length, n):
        result.append(lst[i:i+n])
    return result

