import json
import numpy as np

def getSingleProjectData(proj_list, target_project):

    total_file = 'total'

    prefix = []
    postfix = []
    label = []

    for proj in proj_list:
        
        if proj == target_project or proj == total_file: continue

        print('Getting data for \"' + target_project + '\" from \"' + proj + '\"')

        with open('../data/' + proj, 'r') as f:
            lines = [line.rstrip() for line in f]
        
        for line in lines:

            json_data = json.loads(line)

            prefix.append(json_data['prefix'])
            postfix.append(json_data['postfix'])
            label.append(json_data['label-type'][0])
    
    return np.array(prefix), np.array(postfix), np.array(label)

def getInfo():

    max_len = 0
    source_code_tokens = []
    token_choices = []

    with open('../record/max_len', 'r') as f:
        max_len = int(f.readline().rstrip())
    
    with open('../record/source_code_tokens', 'r') as f:
        source_code_tokens = [int(line.rstrip()) for line in f]
    
    with open('../record/token_choices', 'r') as f:
        token_choices = [int(line.rstrip()) for line in f]

    return max_len, source_code_tokens, token_choices