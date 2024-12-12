import re
# from Evol_Instruct.utils.openai_access import call_chatgpt, call_chatgpt_azure
import json
import os

def extract_domain_name(pddl_text):
    # 正则表达式模式匹配(domain后面跟随的名称)
    pattern = r"\(define \(domain\s+(\S+)"
    # 使用re.search查找第一个匹配的项
    match = re.search(pattern, pddl_text)
    if match:
        # 返回匹配的域名
        return match.group(1).replace(')', '')
    else:
        # 如果没有找到匹配项，返回None
        return None


def extract_pddl(text):
    # 正则表达式模式匹配以```pddl开头，以```结尾的文本块
    pattern = r"```pddl\n(.*?)```"
    # 使用re.DOTALL使得点号(.)可以匹配包括换行符在内的任意字符
    matches = re.findall(pattern, text, re.DOTALL)
    # print(text)
    # if matches:
    # 返回第一个匹配的PDDL文本块，假设文档中只有一个PDDL代码块
    pddl = matches[0].replace('```pddl', '').replace('```', '').strip()
    # domain_name = extract_domain_name(pddl)
    # return pddl, domain_name
    return pddl
    # else:
    #     # 如果没有找到匹配项，返回一个空字符串
    #     raise Exception("pddl not found!")



def count_predicates_actions(content):
    # 分割文件内容为多个部分，每部分代表一个区块（例如domain定义、predicates定义、action定义等）
    parts = content.split('(:')

    predicates_count = 0
    actions_count = 0

    for part in parts:
        # 统计predicates的数量
        if part.startswith('predicates'):
            # 计算predicates定义中的每一行，每行代表一个predicate
            predicates_count += part.count('\n') - 1  # 减去初始的定义行

        # 统计actions的数量
        elif part.startswith('action'):
            actions_count += 1

    return predicates_count, actions_count

def extract_actions(content):
    action_names = []  # 用于存储动作名称的列表

    # 打开并读取PDDL文件
    # with open(domain_file_path, 'r') as file:
    #     content = file.read()

    # 分割文件内容以查找动作定义
    parts = content.split('(:action')

    # 遍历所有找到的动作定义（除了第一个分割结果，因为它在第一个(:action之前）
    for part in parts[1:]:  # 第一个元素不包含动作定义，因此从第二个元素开始遍历
        # 动作名称位于(:action之后的第一个单词
        action_name = part.split()[0]  # 提取动作名称
        action_names.append(action_name)  # 将动作名称添加到列表中

    return action_names  # 返回动作名称列表


from pddlgym.parser import PDDLDomainParser
import os
def parse_actions(pddl_domain):
    pid = os.environ.get('PID', '0')
    temp_pddl = f'./temp_{pid}.pddl'
    f = open(temp_pddl, 'w', encoding='utf')
    f.write(pddl_domain)
    f.close()
    try:
        parser = PDDLDomainParser(temp_pddl, operators_as_actions=True, expect_action_preds=False)
    except:
        parser = PDDLDomainParser(temp_pddl, operators_as_actions=False, expect_action_preds=False)

    # parser._parse_domain_operators()
    action_map = {}
    for k, v in parser.operators.items():
        action_map[k] = len(v.params)
    os.remove(temp_pddl)
    return action_map


def parse_predicates(pddl_domain):
    pid = os.environ.get('PID', '0')
    temp_pddl = f'./temp_{pid}.pddl'
    f = open(temp_pddl, 'w', encoding='utf')
    f.write(pddl_domain)
    f.close()
    try:
        parser = PDDLDomainParser(temp_pddl, operators_as_actions=True, expect_action_preds=False)
    except:
        parser = PDDLDomainParser(temp_pddl, operators_as_actions=False, expect_action_preds=False)

    # parser._parse_domain_operators()
    # parser._parse_domain_predicates()
    predicate_map = {}
    for k, v in parser.predicates.items():
        predicate_map[k] = v.arity
    os.remove(temp_pddl)
    return predicate_map