import time
from copy import deepcopy
import logging
import pickle
import pandas as pd
import math
import random
from collections import defaultdict
from scipy.sparse import csr_matrix
from datetime import datetime
from collections import Counter
import json
import re
from tqdm import tqdm
from typing import *
from tqdm import tqdm
import numpy as np
from numpy import ndarray
import torch
# from torch import Tensor, device
import transformers
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import argparse
from grailqa_prompt_template import *

import pdb

VAR_PREFIX = 'var'
CONSTRAINT_FLAG = '>>> start constraint on'
END_CONSTRAINT_FLAG = '<<< end constraint on'
VERSION = None
META_RELATION = set(['common.topic.article', 'imdb.topic.title_id', 'common.topic.notable_types', 'common.topic.description', 'common.topic.notable_for', 'common.topic.topic_equivalent_webpage', 'kg.object_profile.prominent_type', 'type.object.type', 'common.topic.webpage', 'type.object.key', 'common.webpage.topic', 'type.type.instance'])

def load_jsonl(path):
    with open(path, 'r') as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]
    print("Load %d data from %s" % (len(data), path))
    return data

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    print("Load %d data from %s" % (len(data), path))
    return data

def is_ent(ent_str):
    str_pattern = re.compile(r"^[mg]\..+$")
    if type(ent_str) is not bool and str_pattern.match(ent_str):
        return True
    else:
        return False

def is_rel(ent_str):
    if is_ent(ent_str):
        return False
    else:
        str_pattern = re.compile(r"\w+\.\w+")
        if type(ent_str) is not bool and str_pattern.match(ent_str):
            return True
        else:
            return False

class Command(object):
    def __init__(self, cmd_str):
        self.parse_command(cmd_str)
        
    def __str__(self):
        if self.func_name != 'end':
            return self.return_name + " = " + self.func_name + "(" + ", ".join(self.params) + ")"
        else:
            return self.return_name + " = " + self.params[0]
    
    def parse_command(self, cmd_str):
        if 'obtain_kg_information' in cmd_str:
            func_params = cmd_str[cmd_str.index('(')+1 : cmd_str.index(')')].split(', ')
            self.func_name = 'obtain_kg_information'
            self.return_name = ''
            self.params = func_params
            self.entity = func_params[0]
        elif 'linked_entity' in cmd_str:
            func_params = cmd_str.split(' = ')[1].strip(' ;\n')
            self.func_name = 'linked_entity'
            self.return_name = ''
            self.params = func_params
            self.entity = []
        elif 'answer_type' in cmd_str:
            func_params = cmd_str.split(' = ')[1].strip(' ;\n')
            self.func_name = 'answer_type'
            self.return_name = ''
            self.params = func_params
            self.entity = []
        elif 'ans = ' in cmd_str:
            eles = cmd_str.split(' = ')
            assert len(eles) == 2, cmd_str
            answer_name = eles[1].strip(' \n')
            answer_name = answer_name[answer_name.index('(')+1 : answer_name.index(')')]
            self.func_name = 'end'
            self.return_name = 'ans'
            self.params = [answer_name]
            self.answer = answer_name
        else:
            eq_pos = cmd_str.index('=')
            var_name = cmd_str[:eq_pos-1]
            left_pos = cmd_str.index('(')
            name = cmd_str[eq_pos + 2: left_pos] # get the func name
            func_params = cmd_str[left_pos+1:-1] # get the func paras
            args = func_params.split(', ')
            self.func_name = name
            self.return_name = var_name
            self.params = args
            if name == 'get_head_entity':
                assert len(args) == 2
                self.entity=args[0]
                self.relation = args[1]
            elif name == 'get_tail_entity':
                assert len(args) == 2
                self.entity=args[0]
                self.relation = args[1]
            elif name == 'get_entity_through_type':
                assert len(args) == 1
                self.entity=args[0]
            elif name == 'intersect':
                assert len(args) >= 2, f"{name}({', '.join(args)})"
                self.entity_list = args
            elif name == 'constraint':
                assert len(args) >= 3
                if len(args) == 3:
                    self.func_name = 'superlative'
                    self.entity = args[0]
                    self.relation = args[1]
                    self.operation = args[2]
                else:
                    self.func_name = 'filter'
                    self.entity = args[0]
                    self.relation = args[1]
                    self.operation = args[2]
                    self.value = ', '.join(args[3:])
                    self.params = [self.entity, self.relation, self.operation, self.value]
            elif name == 'count':
                assert len(args) == 1
                self.entity = args[0]
            elif name == 'end':
                assert len(args) == 1
                self.answer = args[0]
            else:
                print(name)
                raise NotImplementedError

class KG_Data(object):
    def __init__(self, sparse_triples_path, sparse_ent_type_path, ent2id_path, rel2id_path):
        triples_path, ent_type_path = sparse_triples_path, sparse_ent_type_path
        print("The sparse data instantiate via int triples from the %s" % (triples_path))
        self._load_triple_and_entity(triples_path=triples_path, ent_type_path=ent_type_path)
        print("Load triple and entity")
        self.ent2id = self._load_pickle_file(ent2id_path)
        print("Load ent2id")
        self.id2ent = self._reverse_dict(self.ent2id)
        print("Load id2ent")
        self.rel2id = self._load_pickle_file(rel2id_path)
        print("Load rel2id")
        self.id2rel = self._reverse_dict(self.rel2id)
        print("Load id2rel")
        print("The sparse KG instantiate over, all triples: %d, max head id: %d." % (
        self.E, self.max_head))
    
    def _load_triple_and_entity(self, triples_path, ent_type_path):
        self.triple = self._load_npy_file(triples_path)
        self.ent_type = self._load_npy_file(ent_type_path)
        self.bin_map = np.zeros_like(self.ent_type, dtype=np.int32)
        self.E = self.triple.shape[0]
        self.head2fact = csr_matrix(
            (np.ones(self.E), (self.triple[:, 0], np.arange(self.E)))).astype('bool')
        self.rel2fact = csr_matrix(
            (np.ones(self.E), (self.triple[:, 1], np.arange(self.E)))).astype('bool')
        self.tail2fact = csr_matrix(
            (np.ones(self.E), (self.triple[:, 2], np.arange(self.E)))).astype('bool')
        self.max_head = max(self.triple[:, 0])
        self.max_tail = max(self.triple[:, 2])
    
    @staticmethod
    def _load_npy_file(filename):
        return np.load(filename)
    
    @staticmethod
    def _load_pickle_file(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _reverse_dict(ori_dict):
        reversed_dict = {v: k for k, v in ori_dict.items()}
        return reversed_dict

VERY_LARGT_NUM = 10 ** 8
PATH_CUTOFF = 10 ** 6
NODE_CUTOFF = 10 ** 4

class KnowledgeGraphSparse(object):
    def __init__(self, kg):
        self.data = kg

    def _fetch_forward_triple(self, seed_set):
        seed_set = np.clip(seed_set, a_min=0, a_max=self.data.max_head)
        indices = self.data.head2fact[seed_set].indices
        return self.data.triple[indices]

    def _fetch_backward_triple(self, seed_set):
        seed_set = np.clip(seed_set, a_min=0, a_max=self.data.max_tail)
        indices = self.data.tail2fact[seed_set].indices
        return self.data.triple[indices]

    def get_relations(self, head_set, reverse=False):
        if reverse:
            triples = self._fetch_backward_triple(head_set)
        else:
            triples = self._fetch_forward_triple(head_set)
        return np.unique(triples[:, 1])

    def get_tails(self, head_set, relation, reverse=False):
        if reverse:
            triples = self._fetch_backward_triple(head_set)
        else:
            triples = self._fetch_forward_triple(head_set)
        rel_indices = (triples[:, 1] == relation)
        triples = triples[rel_indices]
        if len(triples) != 0:
            if reverse:
                return np.unique(triples[:, 0])
            else:
                return np.unique(triples[:, 2])
        else:
            return np.array([])

    def get_triples_along_relation(self, head_set, relation):
        triples = self._fetch_forward_triple(head_set)
        rel_indices = (triples[:, 1] == relation)
        triples = triples[rel_indices]
        return triples
    
    def get_triples(self, head_set, reverse):
        if reverse:
            triples = self._fetch_backward_triple(head_set)
        else:
            triples = self._fetch_forward_triple(head_set)
        return triples

class KG_Execute_Engine(object):
    def __init__(self, kg_data, cid2cstr, strict_mode=False):
        self.sparse_kg = KnowledgeGraphSparse(kg_data)
        self.cid2cstr =  cid2cstr
        self.all_class_ids = list(self.cid2cstr.keys())
        self.strict_mode = strict_mode
    
    def get_tails(self, src_id, rel_id, reverse=False):
        """return list of tail entity ids if not reverse."""
        tails_set = set()
        res_ary = self.sparse_kg.get_tails(src_id, rel_id, reverse)
        tails_list = res_ary.tolist()
        return tails_list
    
    def get_relations(self, src_id, reverse=False):
        """return the list of one-hop relations ids from src_id if not reverse."""
        relations_id_ary = self.sparse_kg.get_relations(src_id, reverse)
        relations_list = relations_id_ary.tolist()
        return relations_list

    def get_triples(self, src_id, reverse=False):
        """return the array of one-hop triple ids from src_id if not reverse."""
        triples = self.sparse_kg.get_triples(src_id, reverse)
        return triples
    
    def find_lc(self, s1, s2): 
        # 生成0矩阵，为方便后续计算，比字符串长度多了一列
        m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)] 
        mmax = 0   # 最长匹配的长度
        p = 0  # 最长匹配对应在s1中的最后一位
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    m[i+1][j+1] = m[i][j] + 1
                    if m[i+1][j+1] > mmax:
                        mmax = m[i+1][j+1]
                        p = i+1
        return (mmax, s1[p-mmax:p])   # 返回最长子串及其长度

    def bind_relation(self, candidates, relation, class_flag=False):
        """bind the relation string to the candidate relations id and return the binded relation id"""
        # special process for .from_to
        if '.from_to' in relation:
            relation_ = relation.replace('.from_to', '.from')
            if relation_ not in [self.sparse_kg.data.id2rel[c] for c in candidates]:
                relation_ = relation.replace('.from_to', '.to')
            relation = relation_

        if relation in [self.sparse_kg.data.id2rel[c] for c in candidates]:
            relation_id = self.sparse_kg.data.rel2id[relation]
            return relation_id
        else:
            return None
        
    def bind_class_type(self, t):
        """bine the tail string to the class string according to all the candidate class types."""
        if t in cstr2cid:
            return cstr2cid[t]
        else:
            return None

    def process_date_value(self, date_str):
        if 't' in date_str:
            date_str = date_str.split('t')[0]
        elif 'T' in date_str:
            date_str = date_str.split('T')[0]

        flag = False
        if date_str.startswith('-'):
            flag = True
            date_str = date_str[1:]
            if len(date_str.split('-')[0]) > 4:
                print(f'Process {date_str} as date value occurring errors!')
                return None
            elif len(date_str.split('-')[0]) == 3:
                date_str = '0' + date_str
            elif len(date_str.split('-')[0]) == 2:
                date_str = '00' + date_str
            elif len(date_str.split('-')[0]) == 1:
                date_str = '000' + date_str

        try:
            if len(date_str.split('-')) == 3:
                date_str = datetime.strptime(date_str, '%Y-%m-%d')
            elif len(date_str.split('-')) == 2:
                date_str = datetime.strptime(date_str, '%Y-%m')
            elif len(date_str.split('-')) == 1:
                date_str = datetime.strptime(date_str, '%Y')
            else:
                print(f'Process {date_str} as date value occurring errors!')
                return None
            return (flag, date_str)
        except Exception as e:
            print(f'Process {date_str} as date value occurring errors!')
            logging.exception(e)
            return None

    def judge_argument_type(self, r_id, arg_list):
        # judge the type of argument according to the arg candidates and relation
        tails = arg_list
        flag = np.zeros(len(tails)) # string:0 int:1 float:2 date:3 dateTime:4 gYear:5 gYearMonth:6
        for idx, t in enumerate(tails):
            if '<int' in t:
                flag[idx] = 1
            elif '<float' in t:
                flag[idx] = 2
            elif any([ _ in t for _ in ['<date>', '<dateTime>', '<gYear>', '<gYearMonth>']]):
                flag[idx] = 3
        
        flag = Counter(flag)
        flag = max(flag, key=lambda x: flag[x])
        if flag == 0:
            arg_type = 'str'
        elif flag == 1:
            arg_type = 'int'
        elif flag == 2:
            arg_type = 'float'
        elif flag == 3:
            arg_type = 'time'

        return arg_type

    def normalize_ent(self, value):
        if '^^' in value:
            value = value.split('^^')[0].strip('"').lower().replace('_',' ')
        elif '@' in value:
            value = value.split('@')[0].strip('"').lower().replace('_',' ')
        else:
            value = value.strip('"').lower().replace('_',' ')
        return value

    def get_compare_param(self, h_id, r_id, arg_list, value, ent_map, operation):
        arg_type = self.judge_argument_type(r_id, arg_list)
        # get value
        value = self.normalize_ent(value)
        if arg_type == 'time' and len(value) == 4 and operation == '=':
            spe_time_filter = True
            val_from = f"{value}-01-01"
            val_from = self.process_date_value(val_from)[-1]
            val_to = f"{value}-12-31"
            val_to = self.process_date_value(val_to)[-1]
        else:
            spe_time_filter = False
            try:
                if arg_type == 'int':
                    value = int(value)
                elif arg_type == 'float':
                    value = float(value)
                elif arg_type == 'time':
                    value = self.process_date_value(value)[-1]
                else:
                    value = value
            except:
                print("Get the value incorrectly.")
                return []
            
        satisfied_arg_list = []
        for t in arg_list:
            t_nor = self.normalize_ent(t)
            if arg_type == 'time':
                try:
                    t_v = self.process_date_value(t_nor)[-1]
                except:
                    continue
                if t_v is None:
                    continue
            elif arg_type == 'int':
                try:
                    t_v = int(t_nor)
                except:
                    continue
            elif arg_type == 'float':
                try:
                    t_v = float(t_nor)
                except:
                    continue
            else:
                t_v = t_nor
            
            if spe_time_filter and val_from <= t_v <= val_to:
                satisfied_arg_list.append(t)
            elif operation == '=' and t_v == value:
                satisfied_arg_list.append(t)
            elif operation == '<=' and t_v <= value:
                satisfied_arg_list.append(t)
            elif operation == '<' and t_v < value:
                satisfied_arg_list.append(t)
            elif operation == '>=' and t_v >= value:
                satisfied_arg_list.append(t)
            elif operation == '>' and t_v > value:
                satisfied_arg_list.append(t)
        satisfied_arg_list = list(set(satisfied_arg_list))
        return satisfied_arg_list, arg_type

    def get_arg_param(self, h_id, r_id, arg_list, is_max=True):
        arg_type = self.judge_argument_type(r_id, arg_list)

        new_arg_list = []
        for t in arg_list:
            t_nor = self.normalize_ent(t)

            if arg_type == 'time':
                time_const = self.process_date_value(t_nor)
                if time_const is not None:
                    new_arg_list.append((t, time_const))
            elif arg_type == 'int':
                new_arg_list.append((t, int(t_nor)))
            elif arg_type == 'float':
                new_arg_list.append((t, float(t_nor)))
            else:
                new_arg_list.append((t, t_nor))

        if arg_type == 'time':
            def takeLast(elem):
                return elem[-1][-1]
            pos_time = []
            neg_time = []
            for al in new_arg_list:
                if al[-1][0]:
                    neg_time.append(al)
                else:
                    pos_time.append(al)

            if is_max:
                pos_time.sort(key=takeLast, reverse=True)
                neg_time.sort(key=takeLast, reverse=False)
                pos_time.extend(neg_time)
                new_arg_list = pos_time
            else:
                pos_time.sort(key=takeLast, reverse=False)
                neg_time.sort(key=takeLast, reverse=True)
                neg_time.extend(pos_time)
                new_arg_list = neg_time
            
            arg_param = new_arg_list[0][0]
        else:
            def takeLast(elem):
                return elem[-1]
            
            if is_max:
                new_arg_list.sort(key=takeLast, reverse=True)
            else:
                new_arg_list.sort(key=takeLast)

            arg_param = new_arg_list[0][0]

        return arg_param

    def filter_with_ans_class_type(self, cand_ids, ans_cand_ids):
        if len(cand_ids & ans_cand_ids) > 0:
            cand_ids = list(cand_ids & ans_cand_ids)
        else:
            cand_ids = list(cand_ids)
        return cand_ids

    def initial_progress(self, ent_map, ans_class_type):
        self.inter_vars = {} # save the entity id of the intermediate variables
        self.binded_entity = set()  # need to exclude
        self.answers = []
        # first obtain the candidates answers set
        if len(ans_class_type) == 0:
            self.ans_cand_ids = set([])
        elif ans_class_type in all_object_type and ans_class_type != 'common.topic':
            t_id = self.sparse_kg.data.ent2id[ans_class_type] # get the sparse id
            r_id = self.sparse_kg.data.rel2id['type.object.type']
            self.ans_cand_ids = set(self.get_tails(t_id, r_id, reverse=True)) # get the head entity ids
        else:
            self.ans_cand_ids = set([])
        self.ent_map = ent_map

    def execute_program(self, cmd):
        ori_cmd = cmd
        # special update entity operation
        if '(' not in cmd and ')' not in cmd and all([v.startswith('var_') for v in cmd.split(' = ')]):
            v1, v2 = cmd.split(' = ')[0].strip(' ;\n'), cmd.split(' = ')[1].strip(' ;\n')
            self.inter_vars[v1] = self.inter_vars[v2]
            return {}, ori_cmd
        # reason with programs
        cmd = Command(cmd)
        if cmd.func_name == 'obtain_kg_information': # obtain candidate information from KG
            h_str = cmd.entity
            return_info = {}
            if h_str in self.inter_vars:
                h_id = self.inter_vars[h_str] # the h is a intermediate variable generated by before command
            else:
                # TODO: Suppose only can bind to mid entity.
                # h_ori = self.bind_entity(h_str, self.ent_map, property_flag=False) # get the mid of the string entity
                h_ori = h_str if is_ent(h_str) else None
                self.binded_entity.add(h_ori)
                if h_ori is None:
                    raise NotImplementedError("No linked entity on the KG in obtain_kg_information. Can't continue reasoning!")
                h_id = self.sparse_kg.data.ent2id[h_ori] # get the id
            out_rel_ids = set(self.get_relations(h_id, reverse=False))
            in_rel_ids = set(self.get_relations(h_id, reverse=True))
            out_rel_strs = [self.sparse_kg.data.id2rel[r] for r in out_rel_ids]
            in_rel_strs = [self.sparse_kg.data.id2rel[r] for r in in_rel_ids]
            return_info['func_name'] = cmd.func_name 
            return_info['return_name'] = 'kg_information'
            return_info["return_value"] = {h_str: {'out_rels': out_rel_strs, 'in_rels': in_rel_strs}}
            return return_info, ori_cmd
        elif cmd.func_name == 'obtain_answer_type_information':
            return {'func_name': cmd.func_name, 'return_name': cmd.return_name, 'return_value': self.all_class_ids}, ori_cmd
        elif cmd.func_name == 'linked_entity':
            return {'func_name': cmd.func_name, 'return_name': cmd.return_name, 'return_value': []}, ori_cmd
        elif cmd.func_name == 'answer_type':
            return {'func_name': cmd.func_name, 'return_name': cmd.return_name, 'return_value': []}, ori_cmd
        elif cmd.func_name == 'get_entity_through_type': 
            t_str = cmd.entity
            r_str = 'type.object.type'
            t_ori = self.bind_class_type(t_str)
            ori_cmd = ori_cmd.replace(t_str, t_ori)
            t_id = self.sparse_kg.data.ent2id[t_ori] # get the sparse id
            r_id = self.sparse_kg.data.rel2id['type.object.type']
            results = self.get_tails(t_id, r_id, reverse=True) # get the head entity ids
            
            if len(results) == 0:
                raise NotImplementedError(f"No target entity on the KG after exectuting the command - [{cmd}]. Can't continue reasoning!")
            else:
                self.inter_vars[cmd.return_name] = results # save the intermediate results
                return {'func_name': cmd.func_name, 'return_name': cmd.return_name, 'return_value': [self.sparse_kg.data.id2ent[e] for e in self.inter_vars[cmd.return_name]]}, ori_cmd
        elif cmd.func_name in ['get_head_entity','get_tail_entity']: # get entity
            h_str = cmd.entity
            r_str = cmd.relation
            reverse = True if cmd.func_name == 'get_head_entity' else False

            if h_str in self.inter_vars:
                h_id = self.inter_vars[h_str] # the h is a intermediate variable generated by before command
            else:
                # TODO: Suppose only can bind to mid entity.
                # h_ori = self.bind_entity(h_str, self.ent_map, property_flag=False) # get the mid of the string entity
                h_ori = h_str if is_ent(h_str) else None
                self.binded_entity.add(h_ori)
                if h_ori is None:
                    raise NotImplementedError(f"No linked entity on the KG with the command - [{cmd}]. Can't continue reasoning!")
                h_id = self.sparse_kg.data.ent2id[h_ori] # get the id
            
            # get the out or in triples of the current entity according to the reverse flag
            current_triples_id = self.get_triples(h_id, reverse)
            # get the out relations id of the current entity
            current_cand_r_ids = np.unique(current_triples_id[:, 1])
            current_bind_r_id = self.bind_relation(current_cand_r_ids, r_str) # get the binded relation
            if current_bind_r_id is None:
                # raise Exception("No linked relation!")
                raise NotImplementedError(f"No linked relation on the KG with the command - [{cmd}]. Can't continue reasoning!")
            # get final result
            valid_t_id = np.unique(current_triples_id[(current_triples_id[:, 1] == current_bind_r_id)][:, 0]) if reverse else np.unique(current_triples_id[(current_triples_id[:, 1] == current_bind_r_id)][:, 2])
            results = valid_t_id.tolist()
            if len(results) == 0:
                raise NotImplementedError(f"No target entity on the KG after exectuting the command - [{cmd}]. Can't continue reasoning!")
            else:
                self.inter_vars[cmd.return_name] = self.filter_with_ans_class_type(set(results), self.ans_cand_ids) # save the intermediate results
                return {'func_name': cmd.func_name, 'return_name': cmd.return_name, 'return_value': [self.sparse_kg.data.id2ent[e] for e in self.inter_vars[cmd.return_name]], 'gold_relation': cmd.relation}, ori_cmd
        elif cmd.func_name == 'intersect':
            ent_set_list = cmd.entity_list
            ent_set_list = [self.inter_vars[e] if e in self.inter_vars else [] for e in ent_set_list]
            ent_set_list = [set(e) for e in ent_set_list if len(e) > 0]
            
            ent_set_join = set()
            for e in ent_set_list:
                if len(ent_set_join) == 0:
                    ent_set_join = e
                else:
                    ent_set_join = ent_set_join & e
                    
            if len(ent_set_join) == 0:
                raise NotImplementedError(f"No target entity on the KG after exectuting the command - [{cmd}]. Can't continue reasoning!")
            else:
                self.inter_vars[cmd.return_name] = self.filter_with_ans_class_type(ent_set_join, self.ans_cand_ids)
                return {'func_name': cmd.func_name, 'return_name': cmd.return_name, 'return_value': [self.sparse_kg.data.id2ent[e] for e in self.inter_vars[cmd.return_name]]}, ori_cmd
        elif cmd.func_name == 'superlative':
            h_str = cmd.entity
            if h_str in self.inter_vars:
                h_id = self.inter_vars[h_str]
            else:
                raise NotImplementedError(f"The entity argument in superlative command - [{cmd}] should be intermediate varint!")
            r_str = [cmd.relation] # list of relation
            binded_r_id = []
            current_ent_id = h_id
            success_flag = True
            triples_id_np_for_each_hop_list = [] 
            for r_ in r_str:
                reverse = False
                # get the out or in triples of the current entity according to the reverse flag
                current_triples_id = self.get_triples(current_ent_id, reverse)
                # get the out relations id of the current entity
                current_cand_r_ids = np.unique(current_triples_id[:, 1])
                current_bind_r_id = self.bind_relation(current_cand_r_ids, r_) # get the binded relation
                if current_bind_r_id is None:
                    success_flag = False
                    if self.strict_mode:
                        raise NotImplementedError(f"No linked relation on the KG with the command - [{cmd}]. Can't continue reasoning!")
                    else:
                        self.inter_vars[cmd.return_name] = h_id
                        return {'func_name': cmd.func_name, 'return_name': cmd.return_name, 'return_value': [self.sparse_kg.data.id2ent[e] for e in self.inter_vars[cmd.return_name]], 'gold_relation': cmd.relation}
                # add the binded relation id to the list
                binded_r_id.append(current_bind_r_id)
                # get the matched triples
                current_triples_id = current_triples_id[(current_triples_id[:, 1] == current_bind_r_id)]
                triples_id_np_for_each_hop_list.append((reverse, current_triples_id))
                # update the current entity
                current_ent_id = np.unique(current_triples_id[:, 0]) if reverse else np.unique(current_triples_id[:, 2])
            if success_flag:
                # the current_ent is the final result
                is_max = True if 'max' in cmd.operation else False
                current_ent_ori = [self.sparse_kg.data.id2ent[eid] for eid in current_ent_id]
                arg_param = self.get_arg_param(h_id, binded_r_id[-1], current_ent_ori, is_max=is_max) # obtain the argmax head entity set of the given relation chain
                arg_param_id = self.sparse_kg.data.ent2id[arg_param]
                current_valid_eid = [arg_param_id]
                for current_triples_id in reversed(triples_id_np_for_each_hop_list):
                    reverse, current_triples_id = current_triples_id
                    new_valid_eid = set()
                    for eid in current_valid_eid:
                        if reverse:
                            tmp = current_triples_id[(current_triples_id[:, 0] == eid)][:, 2].tolist()
                        else:
                            tmp = current_triples_id[(current_triples_id[:, 2] == eid)][:, 0].tolist()
                        new_valid_eid.update(tmp)
                    current_valid_eid = list(new_valid_eid)
                results = current_valid_eid
                if len(results) == 0:
                    if self.strict_mode:
                        raise NotImplementedError(f"No target entity on the KG after exectuting the command - [{cmd}]. Can't continue reasoning!")
                    else:
                        self.inter_vars[cmd.return_name] = self.filter_with_ans_class_type(set(h_id), self.ans_cand_ids)
                        return {'func_name': cmd.func_name, 'return_name': cmd.return_name, 'return_value': [self.sparse_kg.data.id2ent[e] for e in self.inter_vars[cmd.return_name]], 'gold_relation': cmd.relation}, ori_cmd
                else:
                    self.inter_vars[cmd.return_name] = self.filter_with_ans_class_type(set(results), self.ans_cand_ids)
                    return {'func_name': cmd.func_name, 'return_name': cmd.return_name, 'return_value': [self.sparse_kg.data.id2ent[e] for e in self.inter_vars[cmd.return_name]], 'gold_relation': cmd.relation}, ori_cmd
        elif cmd.func_name == 'filter':
            h_str = cmd.entity
            if h_str in self.inter_vars:
                h_id = self.inter_vars[h_str]
            else:
                raise NotImplementedError(f"The entity argument in compare command - [{cmd}] should be intermediate varint!")
            r_str = [cmd.relation] # list of relation
            binded_r_id = []
            current_ent_id = h_id
            success_flag = True
            triples_id_np_for_each_hop_list = [] 
            for r_ in r_str:
                reverse = False
                # get the out or in triples of the current entity according to the reverse flag
                current_triples_id = self.get_triples(current_ent_id, reverse)
                # get the out relations id of the current entity
                current_cand_r_ids = np.unique(current_triples_id[:, 1])
                current_bind_r_id = self.bind_relation(current_cand_r_ids, r_) # get the binded relation
                if current_bind_r_id is None:
                    success_flag = False
                    if self.strict_mode:
                        raise NotImplementedError(f"No linked relation on the KG with the command - [{cmd}]. Can't continue reasoning!")
                    else:
                        self.inter_vars[cmd.return_name] = h_id
                        return {'func_name': cmd.func_name, 'return_name': cmd.return_name, 'return_value': [self.sparse_kg.data.id2ent[e] for e in self.inter_vars[cmd.return_name]], 'gold_relation': cmd.relation}, ori_cmd
                # add the binded relation id to the list
                binded_r_id.append(current_bind_r_id)
                # get the matched triples
                current_triples_id = current_triples_id[(current_triples_id[:, 1] == current_bind_r_id)]
                triples_id_np_for_each_hop_list.append((reverse, current_triples_id))
                # update the current entity
                current_ent_id = np.unique(current_triples_id[:, 0]) if reverse else np.unique(current_triples_id[:, 2])

            value = cmd.value.strip(' "()').split('^^')[0]
            operation = cmd.operation

            if success_flag:
                # the current_ent is the final result
                current_ent_ori = [self.sparse_kg.data.id2ent[eid] for eid in current_ent_id]
                if self.sparse_kg.data.id2rel[binded_r_id[-1]] == 'type.object.type':
                    compare_param = [self.bind_class_type(value)]
                    ori_val = cmd.value
                    rep_val = f'"{compare_param[0]}"'
                    ori_cmd = ori_cmd.replace(ori_val, rep_val)
                else:
                    compare_param, arg_type = self.get_compare_param(h_id, binded_r_id[-1], current_ent_ori, value, self.ent_map, operation) # obtain the argmax head entity set of the given relation chain
                    if operation == '=' and len(compare_param) == 1 and '.from' not in cmd.relation and '.to' not in cmd.relation:
                        ori_val = cmd.value
                        rep_val = f'{compare_param[0]}##{arg_type}'
                        ori_cmd = ori_cmd.replace(ori_val, rep_val)
                    else:
                        ori_val = cmd.value
                        rep_val = f'{ori_val}##{arg_type}'
                        ori_cmd = ori_cmd.replace(ori_val, rep_val)
                compare_param_id = [self.sparse_kg.data.ent2id[i] for i in compare_param]
                current_valid_eid = compare_param_id
                for current_triples_id in reversed(triples_id_np_for_each_hop_list):
                    reverse, current_triples_id = current_triples_id
                    new_valid_eid = set()
                    for eid in current_valid_eid:
                        if reverse:
                            tmp = current_triples_id[(current_triples_id[:, 0] == eid)][:, 2].tolist()
                        else:
                            tmp = current_triples_id[(current_triples_id[:, 2] == eid)][:, 0].tolist()
                        new_valid_eid.update(tmp)
                    current_valid_eid = list(new_valid_eid)
                results = current_valid_eid
                if len(results) == 0:
                    if self.strict_mode:
                        raise NotImplementedError(f"No target entity on the KG after exectuting the command - [{cmd}]. Can't continue reasoning!")
                    else:
                        self.inter_vars[cmd.return_name] = self.filter_with_ans_class_type(set(h_id), self.ans_cand_ids)
                        return {'func_name': cmd.func_name, 'return_name': cmd.return_name, 'return_value': [self.sparse_kg.data.id2ent[e] for e in self.inter_vars[cmd.return_name]], 'gold_relation': cmd.relation}, ori_cmd
                else:
                    self.inter_vars[cmd.return_name] = self.filter_with_ans_class_type(set(results), self.ans_cand_ids)
                    return {'func_name': cmd.func_name, 'return_name': cmd.return_name, 'return_value': [self.sparse_kg.data.id2ent[e] for e in self.inter_vars[cmd.return_name]], 'gold_relation': cmd.relation}, ori_cmd
        elif cmd.func_name == 'count':
            h_str = cmd.entity
            if h_str in self.inter_vars:
                h_id = self.inter_vars[h_str]
            else:
                raise NotImplementedError(f"The entity argument in count command - [{cmd}] should be intermediate varint!")
                return None

            result = len(h_id)
            self.inter_vars[cmd.return_name] = result
            return {'func_name': cmd.func_name, 'return_name': cmd.return_name, 'return_value': [self.inter_vars[cmd.return_name]]}, ori_cmd
        elif cmd.func_name == 'end':
            answer_str = cmd.answer
            assert answer_str in self.inter_vars
            answers = self.inter_vars[answer_str]
            # break
            if type(answers) is list:
                answers = [self.sparse_kg.data.id2ent[a] for a in answers]
                self.answers = list(set(answers) - set(list(str2id.values())) - self.binded_entity)
                # self.answers = list(set(answers))
            elif type(answers) is int:
                self.answers = answers
            return {'func_name': cmd.func_name, 'return_name': "answers", 'return_value': {"answers": self.answers, "binded_entity": self.binded_entity}}, ori_cmd

class Retriever(object):
    def __init__(self, model_name_or_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(self.device)
        print(f"Load model from {model_name_or_path} on {self.device}.")

class LLM(object):
    def __init__(self, agent_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(agent_path)
        self.tokenizer.pad_token = self.tokenizer.unk_token if self.tokenizer.pad_token is not None else self.tokenizer.unk_token
        print(f"Load tokenizer from {agent_path}, whose pad_token is {self.tokenizer.pad_token} and pad_token_id is {self.tokenizer.pad_token_id}.")
        self.model = AutoModelForCausalLM.from_pretrained(agent_path, torch_dtype='auto').to(self.device)
        print(f"Load model from {agent_path} on {self.device}.")
    
    def generate_next_step(self, input):
        inputs = self.tokenizer(input, return_tensors='pt')
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        generated_ids = self.model.generate(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device), do_sample=False, num_beams=1, max_new_tokens=128)
        generated_texts = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        pred = generated_texts.replace(input, '').strip(' \n')
        # print("Input: ", input)
        # print("Prediction pred: ", pred)
        return pred

class Agent(object):
    def __init__(self, KG, mid2str, rel_ret_path, cla_ret_path, agent_path, top_k, pooler, usage, cross_enc, unify_direc):
        self.KG = KG
        self.mid2str = mid2str
        self.usage = usage
        self.cross_enc = cross_enc
        self.top_k = top_k
        self.unify_direc = unify_direc
        if rel_ret_path is not None:
            print('Load relation retriever.')
            self.rel_retriever = Retriever(rel_ret_path)
        if cla_ret_path is not None:
            print('Load class retriever.')
            self.cla_retriever = Retriever(cla_ret_path)
        if agent_path is not None:
            print('Load llm agent.')
            self.llm = LLM(agent_path)
        print("Initialize agent over.")

    def retrieve_rels(self, cands):
        cands_nor, nor2id_map = self.normalize_relation_for_retrieval(cands)
        text_a = [self.question for _ in range(0, len(cands_nor))]
        text_b = cands_nor

        chunk_size = 300
        all_schema_chunks = [(text_a[i: i + chunk_size], text_b[i: i + chunk_size]) for i in range(0, len(cands_nor), chunk_size)]
        assert len(all_schema_chunks)
        scores = []
        
        for chunk in all_schema_chunks:
            try:
                question, rels = chunk[0], chunk[1]
                encodings = self.rel_retriever.tokenizer(question, rels, max_length=128, truncation=True, padding=True, return_tensors='pt')
                if 'token_type_ids' not in encodings:
                    predictions = self.rel_retriever.model(input_ids=encodings['input_ids'].to(self.rel_retriever.device), attention_mask=encodings['attention_mask'].to(self.rel_retriever.device))
                else:
                    predictions = self.rel_retriever.model(input_ids=encodings['input_ids'].to(self.rel_retriever.device), attention_mask=encodings['attention_mask'].to(self.rel_retriever.device),
                                            token_type_ids=encodings['token_type_ids'].to(self.rel_retriever.device))
                score = predictions.logits.detach().cpu().numpy()
                score = list(score[:, 1])
                scores += score
            except Exception as e:
                print('exception' + str(e))
                                
                                
        scores = predictions.logits.detach().cpu().numpy()
        scores = list(scores[:, 1])

        scores = predictions.logits.detach().cpu().numpy()
        scores = list(scores[:, 1])
        scores = np.array(scores)
        predicted_idx = (-scores).argsort()[:100]

        results = [cands_nor[i] for i in predicted_idx]
        results = [r for rel in results[:self.top_k] for r in nor2id_map[rel]]
        return results

    def normalize_entity_type(self, entity_type):
        # entity_type = entity_type.replace('_', ' ').replace('.', ' - ')
        return entity_type
    
    def normalize_relation_for_retrieval(self, relations):
        rel2sim = defaultdict()
        sim2rel = defaultdict(list)
        for rid in relations:
            info = rid_to_info[rid]
            head = info['hn'].lower().strip()
            tail = info['tn'].lower().strip()
            rel = info['rn'].lower().strip()
            if rid.startswith('base.'):
                supp_head = rid.split('.')[1].replace('_',' ')
                rel = f"The relation is {rel} in domain of {supp_head} {head}."
            else:
                rel = f"The relation is {rel} in domain of {head}."
            rel2sim[rid] = rel
            sim2rel[rel].append(rid)
        nor_rels = [rel2sim[r] for r in relations if r in rel2sim]
        return nor_rels, sim2rel

    def construct_start_information(self, question, start_mids, start_entity_type):
        if len(start_mids) > 0:
            start_mids = [(mid, self.mid2str[mid]) for mid in start_mids if mid in self.mid2str]
        input = Q_PROMPT.replace("{question}", question)
        entity_name = []
        for (mid, mstr) in start_mids:
            mstr = '"' + mstr.replace('"', "'") + '"'
            mstr = mstr + "(" + mid + ")"
            entity_name.append(mstr)
        entity_name = "; ".join(entity_name)
        entity_class = []
        for et in start_entity_type:
            et = self.normalize_entity_type(et)
            et = '"' + et + '"'
            entity_class.append(et)
        entity_class = "; ".join(entity_class)
        if len(entity_name) > 0:
            entity_name = TE_PROMPT.replace("{topic_entity}", entity_name)
            input = input + "\n" + entity_name
        if len(entity_class) > 0:
            entity_class = TY_PROMPT.replace("{entity_class_type}", entity_class)
            input = input + "\n" + entity_class
        
        start_info_str = "[BACKGROUND]\n" + input + "\n[/BACKGROUND]"
        start_info_dict = {'question': question, 'entity_name': start_mids, 'entity_class': start_entity_type}
        return start_info_str, start_info_dict
    
    def supplement_schema_linking_programs(self, question, start_mids, id2str, start_entity_type):
        extra_cmds = []
        start_ent_strs = [id2str[mid] for mid in start_mids]
        if len(start_mids) > 0:
            idx = 0
            for str, mid in zip(start_ent_strs, start_mids):
                idx += 1
                tmp = f"linked_entity_{idx} = {mid}[{str}]"
                extra_cmds.append(tmp)
        else:
            tmp = f"linked_entity = none"
            extra_cmds.append(tmp)
        if len(start_entity_type) > 0 and start_entity_type != 'common.topic':
            if start_entity_type in cid2cstr:
                start_entity_type = cid2cstr[start_entity_type]
            else:
                start_entity_type = start_entity_type.replace('_', ' ').replace('.', ' ')
            tmp = f"answer_type = {start_entity_type}"
        else:
            tmp = f"answer_type = none"
        extra_cmds.append(tmp)
        self.history_pros.extend(extra_cmds)
        self.gold_cmds = extra_cmds + self.gold_cmds
    
    def construct_history_information(self):
        history_info = ''
        for his_pro in self.history_pros:
            if CONSTRAINT_FLAG in his_pro or END_CONSTRAINT_FLAG in his_pro:
                history_info = history_info + his_pro + '\n'
            elif 'obtain_kg_information' in his_pro:
                continue
            else:
                pro = his_pro
                if '^^http://www.w3.org/2001/' in pro:
                    value = pro[pro.index('(')+1:pro.index(')')].split(', ')[-1]
                    value_nor = value.split('^^')[0]    
                    pro = pro.replace(value, value_nor)
                his_pro = pro
                history_info = history_info + his_pro + '\n'
        
        if len(history_info) > 0:
            history_info = HI_PROMPT.replace("{history}", history_info.strip('\n'))
        else:
            history_info = HI_PROMPT.replace("{history}", history_info.strip('\n'))
        return history_info

    def build_relation_retrieval_training_data(self, kg_info):
        if kg_info['func_name'] == 'obtain_kg_information':
            self.in_rels = set()
            self.out_rels = set()
            for ent, nei_rels in kg_info['return_value'].items():    
                self.in_rels.update([r for r in nei_rels['in_rels'] if not r.startswith('soft.isbn')])
                self.out_rels.update([r for r in nei_rels['out_rels'] if not r.startswith('soft.isbn')])
        elif kg_info['func_name'] in ['get_head_entity', 'get_tail_entity', 'filter', 'superlative']:
            assert len(self.in_rels) or len(self.out_rels)
            gold_rel = kg_info['gold_relation']
            self.rel_ret_pairs[gold_rel]['in_rels'].update(self.in_rels)
            self.rel_ret_pairs[gold_rel]['out_rels'].update(self.out_rels)
            self.in_rels = set()
            self.out_rels = set()

    def construct_current_kg_relation_information_new(self, kg_info):
        # obtain gold rel if not evaluation
        if self.usage != 'evaluation':
            for pro in self.gold_cmds[len(self.history_pros):]:
                if 'obtain_kg_information' in pro:
                    continue
                elif CONSTRAINT_FLAG in pro or END_CONSTRAINT_FLAG in pro:
                    continue
                else:
                    next_pro = pro
                    break
            cmd = self.parse_pro(next_pro)
            func_name, func_params = cmd['func_name'], cmd['func_params']
            assert func_name in ['get_head_entity','get_tail_entity','superlative','filter']
            gold_relation = func_params[1]
            reverse = True if func_name == 'get_head_entity' else False
            print(f"Next gold cmd: {pro}")

        # process the current kg information
        statements = []
        new_kg_info = deepcopy(kg_info)
        for ent, nei_rels in kg_info['return_value'].items():
            statement = f"For entity {ent}:"

            if self.unify_direc: # only keep the in-relations
                all_r = set()

                o_r = nei_rels['out_rels']
                o_r = [r for r in o_r if not r.startswith('soft.isbn') and r in rid_to_info]
                rev_o_r = [rel_mapping[r] for r in o_r if r in rel_mapping and rel_mapping[r] in rid_to_info]
                all_r.update(o_r)

                i_r = nei_rels['in_rels']
                i_r = [r for r in i_r if not r.startswith('soft.isbn') and r in rid_to_info]
                i_r = set(i_r) - set(rev_o_r)
                all_r.update(i_r)

                all_r = all_r - META_RELATION
            else:
                all_r = set()
                i_r = nei_rels['in_rels']
                i_r = [r for r in i_r if not r.startswith('soft.isbn') and r in rid_to_info]
                all_r.update(i_r)
                o_r = nei_rels['out_rels']
                o_r = [r for r in o_r if not r.startswith('soft.isbn') and r in rid_to_info]
                all_r.update(o_r)
                all_r = all_r - META_RELATION
                
            # perform retrieval if not collect retrieval data
            if self.usage not in ['collect_retrieval', 'ground_pros']:
                ret_r = self.retrieve_rels(all_r)
                i_r = [r for r in i_r if r in ret_r]
                o_r = [r for r in o_r if r in ret_r]
            
            new_kg_info['return_value'][ent]['in_rels'] = i_r
            i_r = " | ".join(i_r)
            new_kg_info['return_value'][ent]['out_rels'] = o_r
            o_r = " | ".join(o_r)
            statement = statement + '\n The out-degree relations: ' + o_r + '.' + '\n The in-degree relations: ' + i_r
            statements.append(statement)
        statements = "\n".join(statements)
        return statements, new_kg_info

    def construct_func_information(self):
        return FU_PROMPT

    def construct_input_prompt(self, kg_info: Dict[str, List]):
        '''using the start_info, history_pros, and kg_info to construct the next step input prompt.'''
        # background
        background = Q_PROMPT_v3.replace('{question}', self.question)
        # kg
        if len(kg_info) > 0:
            if kg_info['func_name'] == 'obtain_kg_information':
                # construct the current KG information
                kg_info_str, new_kg_info = self.construct_current_kg_relation_information_new(kg_info)
                if self.usage == 'collect_retrieval':
                    self.build_relation_retrieval_training_data(kg_info)
            else:
                kg_info_str = ""
                new_kg_info = deepcopy(kg_info)
                if self.usage == 'collect_retrieval' and kg_info['func_name'] in ['get_head_entity', 'get_tail_entity', 'filter', 'superlative']:
                    self.build_relation_retrieval_training_data(kg_info)
        else:
            kg_info_str = ""
            new_kg_info = deepcopy(kg_info)
        if len(kg_info_str) == 0:
            kg_info_str = 'Here is no current KG information.'
        else:
            kg_info_str = KI_PROMPT.replace('{kg_info}', kg_info_str)
        # history
        history_pros = self.construct_history_information()
        # function
        func_info = self.construct_func_information()
        # input
        input = INS_PROMPT.replace('{question}', self.question)
        # format the above information with instruction
        input = IP_PROMPT.replace("{start}", background).replace("{kg}", kg_info_str).replace("{func}", func_info).replace("{end}", input).replace("{history}", history_pros)
        input = TEMPLATE.replace('{{query}}', input)
        input_dict = {'start': background, 'history': deepcopy(self.history_pros), 'kg': new_kg_info}
        return input, input_dict
    
    def parse_pro(self, cmd):
        str = cmd.strip()
        if 'obtain_kg_information' in str:
            left_pos = str.index('(')
            func_params = str[left_pos+1:-1]
            cmd = {"func_name": 'obtain_kg_information', "return_name": '', "func_params": func_params}
        else:
            eq_pos = str.index('=')
            var_name = str[:eq_pos - 1]
            left_pos = str.index('(')
            func_name = str[eq_pos + 2: left_pos] # get the func name
            func_params = str[left_pos+1:-1] # get the func paras
            func_params = func_params.split(', ')
            if func_name == 'constraint':
                assert len(func_params) >= 3
                if len(func_params) == 3:
                    func_name = 'superlative'
                else:
                    func_name = 'filter'
                    new_func_params = [func_params[0], func_params[1], func_params[2], ', '.join(func_params[3:])]
                    func_params = new_func_params
            cmd = {"func_name": func_name, "return_name": var_name, "func_params": func_params}
        return cmd

    def auto_next_step_program(self, input):
        in_constraint = False
        constraint_pros = []
        for pro in self.history_pros:
            if 'obtain_kg_information' in pro:
                continue
            elif CONSTRAINT_FLAG in pro:
                in_constraint = True
                constraint_pros.append(pro)
            elif END_CONSTRAINT_FLAG in pro:
                in_constraint = False
                constraint_pros = []
            elif in_constraint:
                constraint_pros.append(pro)
        new_pro = []
        # already in the last program in the forward constraint process
        # if in_constraint and ('filter(' in constraint_pros[-1] or 'superlative(' in constraint_pros[-1]):
        if in_constraint and 'constraint(' in constraint_pros[-1]:
            # print('constraint_pros: ', constraint_pros)
            cmd = self.parse_pro(constraint_pros[-1])
            cur_var_id = int(cmd['return_name'].strip().split('_')[1].strip())
            for p in reversed(constraint_pros[1:-1]):
                cmd = self.parse_pro(p)
                func_name, return_name, func_params = cmd["func_name"], cmd["return_name"], cmd["func_params"]
                new_func_params = [f"{VAR_PREFIX}_{cur_var_id}"]
                new_func_params.extend(func_params[1:])
                # obtain_kg_info_pro = f"obtain_kg_information({VAR_PREFIX}_{cur_var_id})"
                # new_pro.append(obtain_kg_info_pro)
                cur_var_id += 1
                return_name = f"{VAR_PREFIX}_{cur_var_id}"
                rel = new_func_params[1]
                if self.unify_direc and rel in rel_mapping:
                    new_p = f"{return_name} = {func_name}({', '.join(new_func_params)})"
                    rel_rev = rel_mapping[rel]
                    new_p = new_p.replace(rel, rel_rev)
                else:
                    func_name = 'get_head_entity' if 'get_tail_entity' in func_name  else 'get_tail_entity'
                    new_p = f"{return_name} = {func_name}({', '.join(new_func_params)})"
                new_pro.append(new_p)
            const_var = constraint_pros[0].replace(CONSTRAINT_FLAG, '').strip(' ;\n')
            new_p = f"{const_var} = {VAR_PREFIX}_{cur_var_id}"
            new_pro.append(new_p)
            new_p = constraint_pros[0].replace(CONSTRAINT_FLAG, END_CONSTRAINT_FLAG)
            new_pro.append(new_p)
        return new_pro

    def generate_next_step_program(self, input: str):
        """LLM generates the next step program using the input. When train mode, we directly output the gold program. When test mode, we leverage the LLM output."""
        if self.usage != 'evaluation': # train mode: directly use the gold next step program
            next_pro = []
            # schema_line_count = len([l for l in self.history_pros if l.startswith('linked_entity') or l.startswith('answer_type')])
            # for pro in self.gold_cmds[len(self.history_pros)-schema_line_count:]:
            for pro in self.gold_cmds[len(self.history_pros):]:
                # if pro not in self.history_pros:
                if CONSTRAINT_FLAG in pro:
                    next_pro.append(pro)
                    continue
                next_pro.append(pro)
                break
            return next_pro
        else: # test mode: invoke the LLM to obtain the predicted next step program
            next_pro = self.llm.generate_next_step(input)
            if CONSTRAINT_FLAG in next_pro:
                next_pro = next_pro.split('\nvar')
                next_pro = [next_pro[0], 'var' + next_pro[1]]
            else:
                next_pro = [next_pro]
            return next_pro

    def reason(self, question, str2id, id2str, start_ent_mids, start_ent_cla_type, gold_programs, split):
        self.split = split
        self.question = question
        self.start_ent_mids = start_ent_mids
        self.start_ent_cla_type = start_ent_cla_type
        self.gold_cmds = gold_programs
        self.cur_var_id = 0
        self.trajectory = []
        self.history_pros = []
        self.ground_pros = []
        self.answer = []
        # self.start_information, self.start_information_dict = self.construct_start_information_new(question, start_ent_mids, start_ent_cla_type)
        self.rel_ret_pairs = defaultdict(lambda :defaultdict(set))
        self.cla_ret_pairs = defaultdict(lambda :defaultdict(set))

        self.KG.initial_progress(str2id, start_ent_cla_type[0])

        self.supplement_schema_linking_programs(question, start_ent_mids, id2str, start_ent_cla_type[0])

        # print('---------Begin: \n' + ';\n'.join(self.gold_cmds) + '\n-----\n' + ';\n'.join(self.history_pros))

        return_info = {}
        while True:
            input, input_dict = self.construct_input_prompt(return_info) # the prompt of choosing the next entity
            # print('---------INPUT: \n'+input.replace(FU_PROMPT, ''))
            if self.usage == 'evaluation':
                self.trajectory.append([input])
            else:
                self.trajectory.append([input_dict])
            # judge whether automatically generate the next_pro
            auto_next_pro_list = self.auto_next_step_program(input)
            if len(auto_next_pro_list) != 0:
                next_pro = auto_next_pro_list
                # print('---------next_pro:\n' + '\n'.join(next_pro))
                # self.history_pros.extend(next_pro)
                for p in next_pro:
                    if END_CONSTRAINT_FLAG not in p:
                        print(f'Execute pro {p}')
                        return_info, ground_p = self.KG.execute_program(p)
                        # self.history_pros.append(p)
                        # print('---------Return:')
                        # print(return_info)
                    else:
                        self.history_pros.append(p)
                        self.ground_pros.append(p)
                # print('---------Current: \n' + ';\n'.join(self.gold_cmds) + '\n-----\n' + ';\n'.join(self.history_pros))
            else:
                next_pro = self.generate_next_step_program(input) # generate the next choosed entity command or no kg operation command
                # print('---------next_pro:\n' + '\n'.join(next_pro))
                self.trajectory[-1].append('\n'.join(next_pro))
                self.history_pros.extend(next_pro)
                for p in next_pro:
                    if CONSTRAINT_FLAG not in p:
                        print(f'Execute pro {p}')
                        return_info, ground_p = self.KG.execute_program(p)
                        # print('---------Return:')
                        # print(return_info)
                        self.ground_pros.append(ground_p)
                    else:
                        self.ground_pros.append(p)
                # print('---------Current: \n' + ';\n'.join(self.gold_cmds) + '\n-----\n' + ';\n'.join(self.history_pros))
            if return_info is None:
                break
            if len(self.trajectory) > 30:
                self.answer = []
                raise NotImplementedError("Maxmize 30 steps! Stop reasoning!")
            if len(return_info) > 0 and return_info['return_name'] == 'answers':
                self.answer = return_info['return_value']['answers']
                break
        return self.answer, self.rel_ret_pairs, self.trajectory, self.ground_pros

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str)
    parser.add_argument('--rel_retriever_path', type=str)
    parser.add_argument('--cla_retriever_path', type=str)
    parser.add_argument('--agent_path', type=str)
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--invalid_case_save_path', type=str)
    parser.add_argument('--pooler', type=str)
    parser.add_argument('--topk', type=int)
    parser.add_argument('--version', type=str)
    parser.add_argument('--cross_enc', type=bool)
    parser.add_argument('--usage', type=str, help="Select from ['collect_retrieval', 'collect_trajectory', 'evaluation']")
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--unify_direc', type=bool, default=False)
    args = parser.parse_args()

    if VERSION is None:
        VERSION = args.version
    assert args.usage in ['ground_pros', 'collect_retrieval', 'collect_trajectory', 'evaluation'], args.usage
    split = args.split
    if args.unify_direc:
        print("Unify direction")
    else:
        print("Not unify direction")

    with open('./data/1vN_rels.pickle', 'rb') as f:
        duplicate_rels = pickle.load(f)
    # 加载类型信息
    with open('./data/grailqa/ori_cla2sim_cla_mapping.json','r') as f:
        cid2cstr = json.load(f)
        cstr2cid = {v:k for k,v in cid2cstr.items()}
    # 加载实体类型
    with open('./data/object_type.pickle', 'rb') as f:
        all_object_type = pickle.load(f)
        all_object_type = set(all_object_type)
    # 加载实体
    if 'webqsp' in args.input_path:
        schema_link = f'./data/webqsp/{split}_schema_linking_gold_v0.json'
    elif 'cwq' in args.input_path:
        schema_link = f'./data/cwq/{split}_schema_linking_gold_v0.json'
    elif 'grailqa' in args.input_path:
        schema_link = f'./data/grailqa/{split}_schema_linking_gold_v0.json'
    with open(schema_link, "r") as f:
        schema_link = json.load(f)
    qid2str2id = defaultdict()
    mid2str = {}
    for qid, schema in schema_link.items():
        ent_ori2nor = schema['ent_ori2nor']
        nor2ori = {}
        for ori, nor in ent_ori2nor.items():
            nor2ori[nor.lower()] = ori
            mid2str[ori] = nor.lower()
        qid2str2id[qid] = nor2ori
    print("Totally %d str to id mappings!" % len(qid2str2id))
    print("Totally %d id to str mappings!" % len(mid2str))
    # 加载关系-逆关系 映射
    with open('data/par_rid_reverse_map.json','r') as f:
        rel_mapping = json.load(f)
    print(f"Load {len(rel_mapping)} pairs of relation mappings.")
    # 加载关系的信息
    with open('data/par_rid_to_info.json', 'r') as f:
        rid_to_info = json.load(f)
    # 加载数据
    input_path = args.input_path # f"data/grailqa/{split}_w_pro_v2.json"
    if 'jsonl' in input_path:
        with open(input_path, "r") as f:
            all_data = f.readlines()
            all_data = [json.loads(l) for l in all_data]
    else:
        with open(input_path, "r") as f:
            all_data = json.load(f)
    print(f"Load {len(all_data)} from {input_path}")
    
    # 加载KG
    if args.debug:
        sparse_kg_source_path = "./KB/SparseKG/webqsp_debug/KG_triples.npy"
        sparse_ent_type_path = "./KB/SparseKG/webqsp_debug/ent_type_ary.npy"
        sparse_ent2id_path = "./KB/SparseKG/webqsp_debug/ent2id.pickle"
        sparse_rel2id_path = "./KB/SparseKG/webqsp_debug/rel2id.pickle"
    else:
        # sparse_kg_source_path = "./KB/SparseKG/KG_triples.npy"
        # sparse_ent_type_path = "./KB/SparseKG/ent_type_ary.npy"
        # sparse_ent2id_path = "./KB/SparseKG/ent2id.pickle"
        # sparse_rel2id_path = "./KB/SparseKG/rel2id.pickle"
        sparse_kg_source_path = "./KB/freebase/sparse_kg/KG_triples.npy"
        sparse_ent_type_path = "./KB/freebase/sparse_kg/ent_type_ary.npy"
        sparse_ent2id_path = "./KB/freebase/sparse_kg/ent2id.pickle"
        sparse_rel2id_path = "./KB/freebase/sparse_kg/rel2id.pickle"
    kg = KG_Data(sparse_triples_path=sparse_kg_source_path, sparse_ent_type_path=sparse_ent_type_path, ent2id_path=sparse_ent2id_path, rel2id_path=sparse_rel2id_path)

    KG = KG_Execute_Engine(kg, cid2cstr)
    rel_retriever_path = args.rel_retriever_path
    cla_retriever_path = args.cla_retriever_path
    agent_path = args.agent_path
    pooler = args.pooler
    topk = args.topk
    agent = Agent(KG, mid2str, rel_retriever_path, cla_retriever_path, agent_path, topk, pooler, args.usage, args.cross_enc, args.unify_direc)

    # 计算
    all_f1 = []
    count = 0
    invalid_count = 0
    invlaid_idx = []
    output_path = args.output_path
    with open(output_path, 'w') as f:
        for data in all_data:
            # if data['qid'] not in ['WebQTest-305_01c160c410942d97fa50318448a15951']:
            #     continue
            count += 1
            if args.usage == 'evaluation':
                question = data['question'].lower().strip('\n')
                qid = data['qid']
                start_ent_mids = data['start_ent_mids']
                class_type = data['class_type']
                gold_program = data['programs']
                answers = data['answer']
                answer_entity = [a['answer_argument'] for a in answers]
                str2id = data['str2id']
            else:
                question = data['question'].lower().strip('\n')
                qid = data['qid']
                start_ent_mids = data['start_ent_mids']
                class_type = data['class_type']
                gold_program = data['programs']
                answers = data['answer']
                answer_entity = [a['answer_argument'] for a in answers]
                str2id = qid2str2id[str(qid)]
            try:
                # pdb.set_trace()
                results, rel_ret_pair, trajectory, ground_pros = agent.reason(question, str2id, mid2str, start_ent_mids, [class_type], gold_program.split(';\n'), split)
            except Exception as e:
                invlaid_idx.append(data['qid'])
                invalid_count += 1
                # logging.exception(e)
                print(f'Error: [{qid}]')
                all_f1.append(0.0)
                continue
            # results, rel_ret_pair, trajectory = agent.reason(question, str2id, start_ent_mids, [class_type], gold_program.split(';\n'), split)

            if type(results) is list:
                pred = set(results)
                gold = set(answer_entity)
            else:
                pred = {str(results)}
                gold = set(answer_entity)
            if len(gold) == 0:
                # f1 = 1.0 if len(pred) == 0 else 0.0
                f1 = 1.0
            else:
                if len(pred) == 0:
                    f1 = 0.0
                else:
                    precision = (len(pred & gold) + 0.0) / len(pred)
                    recall = (len(pred & gold) + 0.0) / len(gold)
                    f1 = 0.0 if (precision + recall) == 0 else (2*precision*recall) / (precision+recall)
            all_f1.append(f1)

            if f1 == 0:
                print(f"{qid}: {f1}")
                print(gold_program)

            if args.usage == 'ground_pros':
                data['ground_pros'] = ';\n'.join(ground_pros)
                new_d = data
            elif args.usage == 'collect_retrieval':
                new_rel_ret_pair = defaultdict(lambda :defaultdict(list))
                for r, cands in rel_ret_pair.items():
                    for k, v in cands.items():
                        new_rel_ret_pair[r][k] = list(v)
                new_d = {'qid': qid, 'question': question, 'rel_ret': new_rel_ret_pair}
            elif args.usage == 'collect_trajectory':
                new_d = {'qid':qid, 'question':question, 'trajectory': trajectory}
                if count == 1:
                    for inp_oup in trajectory:
                        if len(inp_oup) == 2:
                            inp, oup = inp_oup[0], inp_oup[1]
                        else:
                            inp = inp_oup[0]
                            oup = 'None'
                        print("Input: ", inp)
                        print("Output: ", oup)
                        print('------------')
            elif args.usage == 'evaluation':
                new_d = {'qid':qid, 'question':question, 'trajectory': trajectory, 'gold_answer': list(gold), 'prediction': list(pred)}
                if count == 1:
                    for inp_oup in trajectory:
                        if len(inp_oup) == 2:
                            inp, oup = inp_oup[0], inp_oup[1]
                        else:
                            inp = inp_oup[0]
                            oup = 'None'
                        print("Input: ", inp)
                        print("Output: ", oup)
                        print('------------')

            f.write(json.dumps(new_d)+"\n")
            if count % 100 == 0:
                print(f"Processed {count} samples, in which have {invalid_count} invalid samples. And current mean F1: {np.mean(all_f1)}")

    with open(args.invalid_case_save_path, 'w') as f:
        json.dump(invlaid_idx, f)
    print(f"Save {len(invlaid_idx)}.")

    print("Totally process prediction %d / %d / %d" % (len(all_f1), invalid_count, len(all_data)))
    print("Average f1: %.4f" % np.mean(all_f1))