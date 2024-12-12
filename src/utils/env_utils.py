import pddlgym
import imageio
from pddlgym_planners.fd import FD
# from nl_map import predicate_map, description_map
import nltk
from pddlgym.structs import Literal, Predicate


def get_action_space(env, last_obs):
    data = [literal_to_text(literal, predicate_map=None) for literal in env.action_space.all_ground_literals(last_obs)]
    return "Valid actions are: " + ", ".join(data)


def literal_to_text(literal, predicate_map):
    if predicate_map is None:
        return literal_to_text_wonl(literal)
    predicate_name = literal.predicate.name
    objects = literal.variables
    predicate_map = {k.lower():v for k, v in predicate_map.items()}
    if predicate_name.lower() in predicate_map:
        predicate_format = predicate_map[predicate_name.lower()]
        if predicate_format[-1] != '.':
            predicate_format += '.'
        objects_name = [str(obj.name) for obj in objects]
        # print(objects_name, predicate_format)
        # text = predicate_format.format(*objects_name)
        text = predicate_format
        for idx, obj in enumerate(objects):
            arg_name = f'arg{idx+1}'
            text = text.replace(f"{{%s}}" % arg_name, obj)
    else: 
        text = predicate_name + " " + " ".join([str(obj.name) for obj in objects]) + '.'
    objects_name = [str(obj) for obj in objects]
    return text

def literal_to_text_wonl(literal):
    predicate_name = literal.predicate.name
    objects = literal.variables
    params = ", ".join([str(obj.name) for obj in objects])
    return f'{predicate_name}({params})'

def get_obs_text_wonl(obs):
    state = obs.literals # conjunction of literals
    state_text = [literal_to_text(literal) for literal in state]
    state_text.sort()
    state_text = " ".join(state_text)
    return state_text

def get_goal_text_wonl(obs):
    goal = obs.goal # conjunction of literals
    goal = [literal_to_text(literal) for literal in goal.literals]
    goal.sort()
    goal_text = "The goal is to satisfy the following conditions: " + " ".join(goal) 
    return goal_text

def get_goal_and_obs(obs, predicate_map):
    goal = obs.goal # conjunction of literals
    goal = [literal_to_text(literal, predicate_map) for literal in goal.literals]
    goal.sort()
    goal_text = "The goal is to satisfy the following conditions: " + " ".join(goal) 
    state = obs.literals # conjunction of literals
    state_text = [literal_to_text(literal, predicate_map) for literal in state]
    state_text.sort()
    state_text = " ".join(state_text)
    return goal_text, state_text, obs.goal.literals


def get_obs_text(obs, predicate_map):
    state = obs.literals # conjunction of literals
    state_text = [literal_to_text(literal, predicate_map) for literal in state]
    state_text.sort()
    state_text = " ".join(state_text)
    return state_text

# uncomment if there is not punkt in your environment
# nltk.download('punkt')
def text_to_action(env, last_obs, text):
    text = text.lower()
    all_valid_actions = env.action_space.all_ground_literals(last_obs)
    all_valid_predicates = [action.predicate for action in all_valid_actions]
    predicates_names = {predicate.name.lower():predicate for predicate in all_valid_predicates}
    predicates_obj_nums = {predicate.name.lower(): predicate.arity for predicate in all_valid_predicates}
    
    all_valid_objects = [obj for obj in last_obs.objects] # objects as class
    all_valid_objects_name = [str(obj) for obj in all_valid_objects] # objects as string
    all_valid_objects_id = [obj.name for obj in all_valid_objects] # objects as id
    # parse text
    tokens = nltk.word_tokenize(text)
    
    predicate_name = None   
    for token in tokens:
        if token in predicates_names:
            predicate_name = token
            break
    
    if predicate_name is None:
        return None
    else:
        predicate_obj_nums = predicates_obj_nums[predicate_name]
        if predicate_obj_nums == 0:
            return Literal(Predicate(predicate_name))
        else:
            # find the objects
            objects = []
            for token in tokens:
                if token in all_valid_objects_name:
                    objects.append(token)
                elif token in all_valid_objects_id:
                    objects.append(all_valid_objects[all_valid_objects_id.index(token)])
                else:
                    continue
            if len(objects) > predicate_obj_nums:
                objects = objects[:predicate_obj_nums]
            elif len(objects) < predicate_obj_nums:
                return None
            else:
                pass
        
    predicate = predicates_names[predicate_name]
    literal = Literal(predicate, objects)
    return literal


'''
{
    "id": xxx,
    "task_name": xxx,
    "goal": xxx,
    "prompt": xxx,
    "trajectory": [{
        "action": xxx,
        "thought": xxx,
        "observation": xxx,
    },...]
}
'''