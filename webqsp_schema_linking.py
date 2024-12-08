import json
from collections import defaultdict
import csv
from tqdm import tqdm
import numpy as np
import random
import argparse
random.seed(42)

TYPE_OBJECT_TYPE='type.object.type'

def process_classes(classes, type):
    if type == 'v0':
        # 不做简化，使用原始表示，但使用id表示class
        ori2nor = defaultdict()
        ori2id = defaultdict()
        id2ori = defaultdict()
        for idx, c in enumerate(classes):
            # c_nor = c.split('.')[-1].replace('_', ' ')
            c_nor = c
            c_id = 'cla_' + str(idx)
            ori2nor[c] = c_nor
            ori2id[c] = c_id
            id2ori[c_id] = c
    elif type == 'v6':
        # 做简化且仅保留最后一段，过滤到_和.，但使用id表示class
        ori2nor = defaultdict()
        ori2id = defaultdict()
        id2ori = defaultdict()
        for idx, c in enumerate(classes):
            c_nor = c.split('.')[-1].replace('_', ' ')
            c_id = 'cla_' + str(idx)
            ori2nor[c] = c_nor
            ori2id[c] = c_id
            id2ori[c_id] = c
    if type == 'v7':
        # 做简化且仅保留最后一段，过滤到_和.，但使用id表示class
        ori2nor = defaultdict()
        ori2id = defaultdict()
        id2ori = defaultdict()
        for idx, c in enumerate(classes):
            c_nor = c.split('.')[-1].replace('_', ' ')
            c_id = 'cla_' + str(idx)
            ori2nor[c] = c_nor
            ori2id[c] = c_id
            id2ori[c_id] = c
        
    return ori2nor, ori2id, id2ori

def process_relations(relations, type):
    if type == 'v0':
        # 不做简化，nor使用原始表示，支持id表示
        ori2nor = defaultdict()
        ori2id = defaultdict()
        id2ori = defaultdict()
        for idx, r in enumerate(relations):
            # r_nor = r.split('.')[-1].replace('_', ' ')
            r_nor = r
            r_id = 'rel_' + str(idx)
            ori2nor[r] = r_nor
            ori2id[r] = r_id
            id2ori[r_id] = r
    elif type == 'v6':
        # nor做简化且仅保留最后一段，过滤到_和.，支持id表示
        ori2nor = defaultdict()
        ori2id = defaultdict()
        id2ori = defaultdict()
        for idx, r in enumerate(relations):
            if r == 'type.object.type':
                r_nor = 'object type'
            else:
                r_nor = r.split('.')[-1].replace('_', ' ')
            r_id = 'rel_' + str(idx)
            ori2nor[r] = r_nor
            ori2id[r] = r_id
            id2ori[r_id] = r
    elif type == 'v7':
        # 去掉特殊字符做简化，支持id表示
        ori2nor = defaultdict()
        ori2id = defaultdict()
        id2ori = defaultdict()
        for idx, r in enumerate(relations):
            r_nor = '<'+r.replace('_', ' ').replace('.', '/')+'>'
            r_id = 'rel_' + str(idx)
            ori2nor[r] = r_nor
            ori2id[r] = r_id
            id2ori[r_id] = r
    return ori2nor, ori2id, id2ori

version = 'v0'
dataset = 'webqsp'
# dataset = 'cwq'
data_split = ['train', 'dev', 'test']

for split in data_split:
    print('Start ', split)

    # save gold schema
    qid2str2id = defaultdict(lambda: defaultdict())
    qid2id2str = defaultdict(lambda: defaultdict())
    qid2gold_rel = defaultdict(list)
    qid2gold_cls = defaultdict(list)
    all_id2str = defaultdict()

    train_path = f'data/{dataset}/{dataset}_0103.{split}.json'
    with open(train_path, 'r') as f:
        all_data = json.load(f)
    print("Load %d train examples" % len(all_data))

    for d in all_data:
        qid = str(d['qid'])
        if '#MANUAL ' in d['sparql_query']:
            continue
        nodes = d['graph_query']['nodes']
        ent_flag = False
        for node in nodes:
            # if node['node_type'] in ['entity', 'literal']:
            if node['node_type'] in ['entity']:
                qid2id2str[qid][node['id']] = node['friendly_name']
                qid2str2id[qid][node['friendly_name']] = node['id']
                ent_flag = True
            # elif node['node_type'] in ['literal']:
            #     print(qid)
            elif node['node_type'] in ['class'] and int(node['question_node']) == 1:
                if node['id'] != 'common.topic':
                    qid2gold_cls[qid].append(node['id'])
        # assert len(qid2gold_cls[qid]) <= 1 
        assert ent_flag
        relations = d['graph_query']['edges']
        for rel in relations:
            qid2gold_rel[qid].append(rel['relation'])
    print("Load %d id mappings from train data!" % len(qid2str2id))

    ent_link = {'id2str':qid2id2str, 'str2id':qid2str2id, 'gold_cls':qid2gold_cls, 'gold_rel':qid2gold_rel}

    # output_path = f'data/{dataset}/{split}_gold_schema.json'
    # with open(output_path, "w") as f:
    #     json.dump(ent_link, f)
    # print("Save the mapping dict to %s" % output_path)

    # with open(f'data/grailqa/{split}_gold_schema.json','r') as f:
        # gold_schema = json.load(f)
    # qid2id2str, qid2str2id, qid2gold_cls, qid2gold_rel = gold_schema['id2str'], gold_schema['str2id'], gold_schema['gold_cls'], gold_schema['gold_rel']
    id2el = qid2id2str
    # print("Load %d gold schema linking results" % len(qid2id2str))

    # path = f"data/grailqa/origin/grailqa_v1.0_{split}.json"
    # with open(path, 'r') as f:
        # all_data = json.load(f)
    # print("Load %d data" % len(all_data))

    total = 0
    d2clues = {}
    # print("Start process %s" % path)
    for i, item in tqdm(enumerate(all_data)):
        total += 1
        new_d = {}
        qid = str(item['qid'])
        question = item['question']
        if '#MANUAL ' in item['sparql_query']:
            print("Skip")
            continue
        # process classes
        gold_cls = qid2gold_cls[qid]
        # process relations
        gold_rel = qid2gold_rel[qid]
        # gold_rel.append(TYPE_OBJECT_TYPE)
        rel_ori2nor, rel_ori2id, rel_id2ori = process_relations(gold_rel, version)
        cla_ori2nor, cls_ori2id, cls_id2ori = process_classes(gold_cls, version)
        new_d['rel_ori2nor'] = rel_ori2nor
        new_d['rel_ori2id'] = rel_ori2id
        new_d['rel_id2ori'] = rel_id2ori
        new_d['cla_ori2nor'] = cla_ori2nor
        new_d['cls_ori2id'] = cls_ori2id
        new_d['cls_id2ori'] = cls_id2ori
        new_d['gold_rel'] = gold_rel
        new_d['gold_cla'] = gold_cls
        
        # process entities
        el = qid2id2str[qid]
        ent_ori2nor, ent_ori2id, ent_id2ori = defaultdict(), defaultdict(), defaultdict()
        for idx, ori_nor in enumerate(el.items()):
            ori, nor = ori_nor
            ori = ori
            nor = nor
            if type(ori) is int:
                print(qid)
                print(ori, ' - ', nor)
            elif '^^' in ori:
                continue
            # if nor.lower() not in question.lower():
                # print(question, ' - ', nor)
            e_id = 'ent_' + str(idx)
            ent_ori2nor[ori] = nor
            ent_ori2id[ori] = e_id
            ent_id2ori[e_id] = ori

        new_d['ent_ori2nor'] = ent_ori2nor
        new_d['ent_ori2id'] = ent_ori2id
        new_d['ent_id2ori'] = ent_id2ori

        d2clues[qid] = new_d

    print("Sample:-------")
    for k, v in new_d.items():
        print(k)
        print(v)
    
    output_path = f'data/{dataset}/{split}_schema_linking_gold_{version}.json'
    with open(output_path, 'w') as f:
        json.dump(d2clues, f)
    print("Save %d data %s" % (len(d2clues), output_path))