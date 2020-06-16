from cfgs.base_cfgs import Cfgs
import json
# from core.data.data_utils import ans_stat
import requests
import time


# extract the dbpedia entities from nl using dbpedia spotlight
def entity_extractor(text, confidence=0.4):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36",
        "Connection": "Keep-Alive",
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate, br"
    }
    url = "http://localhost:2222/rest/annotate?"
    params = {"text": text,
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
split_list = ['val', 'test']
stat_ques_list = []
path_list = ['datasets/vqa2db/ques_entities.json', 'datasets/vqa2db/failed_ques.json']

# init_json(path_list)

for i in split_list:
    stat_ques_list += json.load(open(__C.QUESTION_PATH[i], 'r'))['questions']

# stat_ques_list = \
#             json.load(open(__C.QUESTION_PATH['train'], 'r'))['questions'] + \
#             json.load(open(__C.QUESTION_PATH['val'], 'r'))['questions'] + \
#             json.load(open(__C.QUESTION_PATH['test'], 'r'))['questions'] + \
#             json.load(open(__C.QUESTION_PATH['vg'], 'r'))['questions']

failed_ques = {}
ques_entities = {}
ques_len = len(stat_ques_list)
total = 0

for i, ques in enumerate(stat_ques_list):
    text = ques['question']
    ques_id = ques['question_id']
    r, status = entity_extractor(text)
    if status == 200 and len(r) != 0:
        ques_entities[ques_id] = r
        total += 1
    else:
        failed_ques[ques_id] = ques_id
    if (i + 1) % 10000 == 0:
        print("ques_entity process: {}|{}, success: {}".format(i, ques_len, total))

    if (i + 1) % 100000 == 0:
        json_store(path_list[0], ques_entities)
        json_store(path_list[1], failed_ques)
        ques_entities = {}
        failed_ques = {}

json_store(path_list[0], ques_entities)
json_store(path_list[1], failed_ques)
print('finish ques_entities')
# with open('core/data/ques_entities.json', 'w') as f:
#     f.write(json.dumps(ques_entities, indent=4))


# with open('core/data/failed_ques.json', 'w') as f:
#     f.write(json.dumps(failed_ques, indent=4))
# print("finish ques_entities")
