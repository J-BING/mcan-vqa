from cfgs.base_cfgs import Cfgs
import json
# from core.data.data_utils import ans_stat
import requests
import time

#extract the dbpedia entities from nl using dbpedia spotlight
def entity_extractor(text, confidence=0.4):
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36",
        "Connection": "Keep-Alive",
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate, br"
    }
    url = "http://localhost:2222/rest/annotate?"
    params = {"text":text,
              "confidence": confidence}
    r = requests.get(url=url, params=params, headers=headers)
    if r.status_code == 200:
        return r.json(), r.status_code
    else:
        return {}, r.status_code

def json_store(path, file):
    load_f = json.load(open(path, 'r', encoding='utf-8'))
    with open(path, 'w') as f:
        merge_file = {**load_f, **file}
        f.write(json.dumps(merge_file, indent=4))

def init_json(path_list):
    init = {}
    for path in path_list:
        with open(path, 'w') as f:
            f.write(json.dumps(init, indent=4))
    

__C = Cfgs()
split_list = ['train', 'val']
ans_list = []
path_list = ['datasets/vqa2db/ans_entities.json', 'datasets/vqa2db//failed_ans.json']

# init_json(path_list)

for i in split_list:
    ans_list += json.load(open(__C.ANSWER_PATH[i], 'r'))['annotations']

# ans_list = \
#             json.load(open(__C.ANSWER_PATH['train'], 'r'))['annotations'] + \
#             json.load(open(__C.ANSWER_PATH['val'], 'r'))['annotations'] + \
#             json.load(open(__C.ANSWER_PATH['vg'], 'r'))['annotations']

# ans_to_ix, ix_to_ans = ans_stat('core/data/answer_dict.json')

ans_len = len(ans_list)
total = 0
failed_ans = {}
ans_entities = {}

for i, ans in enumerate(ans_list):
    # begin = time.time()
    text = ans['multiple_choice_answer']
    ques_id = ans['question_id']
    r, status = entity_extractor(text)
    # end = time.time()
    
    if status == 200 and len(r) != 0:
        ans_entities[ques_id] = r
        total += 1
    else:
        failed_ans[ques_id] = ques_id
    
    if (i+1)%10000 == 0:
        print("ans_entity process: {}|{}, success: {}".format(i+1, ans_len, total))

    # print("time cost: {}".format(end-begin))

    if (i+1)%100000 == 0:
        json_store(path_list[0], ans_entities)
        json_store(path_list[1], failed_ans)
        ans_entities = {}
        failed_ans = {}
    # if i == 3:
    #     break    

json_store(path_list[0], ans_entities)
json_store(path_list[1], failed_ans)

# with open(path_list[0], 'w') as f:
#     f.write(json.dumps(ans_entities, indent=4))

# with open(path_list[1], 'w') as f:
#     f.write(json.dumps(failed_ans, indent=4))
print("finish ans_entities")
