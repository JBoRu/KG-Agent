from collections import defaultdict
from tqdm import tqdm
import json
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON
import urllib
import sys
sparql = SPARQLWrapper("http://localhost:3001/sparql")
sparql.setReturnFormat(JSON)
import logging

TYPE_OBJECT_REL = 'type.object.type'
SUFFIX_MAP = {
    'type.float': '^^<http://www.w3.org/2001/XMLSchema#float>',
    'type.int': '^^<http://www.w3.org/2001/XMLSchema#int>',
    'type.datetime': '^^<http://www.w3.org/2001/XMLSchema#date>'
}
PREFIX = '''PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX : <http://rdf.freebase.com/ns/>
'''

def execute_sql(sql):
    answers = set()
    # try:
    sparql.setQuery(sql)
    results = sparql.query().convert()
    # except urllib.error.URLError:
    # except Exception:
        # logging.exception(Exception)
        # return []
    for result in results['results']['bindings']:
        if 'value' in result:
            answers.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))
        elif 'x' in result:
            answers.add(result['x']['value'].replace('http://rdf.freebase.com/ns/', ''))
        else:
            print(sql)
            print(result)
            raise NotImplementedError("New result varibles.")

    return answers

def get_range_type(relation):
    range_type = []

    query2 = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x1 AS ?value) WHERE {
        SELECT DISTINCT ?x1  WHERE {
        """
        ':' + relation + ' rdfs:range' + ' ?x1 . '
        """
        }
        }
        """)
    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        range_type.append(result['value']['value'].split("http://rdf.freebase.com/ns/type.")[1])
    if len(range_type) > 0:
        return range_type[0]
    else:
        # process special cases
        if 'number' in relation:
            return 'float'
        else:
            return ''

def compute_f1(gold, pred):
    if len(gold) == 0:
        # if len(pred) == 0:
        #     f1 = 1.0
        # else:
        #     f1 = 0.0
        f1 = 1.0
    else:
        if len(pred) == 0:
            f1 = 0.0
        else:
            precision = (len(pred & gold) + 0.0) / len(pred)
            recall = (len(pred & gold) + 0.0) / len(gold)
            if (precision + recall) == 0:
                f1 = 0.0
            else:
                f1 = (2*precision*recall) / (precision+recall)
    return f1

def execute_sql_obtain_number_type(sql):
    answers = set()
    sparql.setQuery(sql)
    try:
        results = sparql.query().convert()
    # except urllib.error.URLError:
    except Exception:
        logging.exception(Exception)
        return None
    for result in results['results']['bindings']:
        if 'datatype' in result['value']:
            k = result['value']['datatype']
            v = result['value']['value'].replace('http://rdf.freebase.com/ns/', '')
            answers.add((k,v))
        elif 'type' in result['value']:
            if 'xml:lang' in result['value'] and result['value']['xml:lang'] == 'en':
                k = 'literal'
                v = result['value']['value'].replace('http://rdf.freebase.com/ns/', '')
                answers.add((k,v))
            elif 'xml:lang' not in result['value']:
                k = 'literal'
                v = result['value']['value'].replace('http://rdf.freebase.com/ns/', '')
                answers.add((k,v))
        else:
            raise NotImplementedError("New type in result.")

    return answers

def normalize_number_or_date(number, relation):
    if '^^' in number:
        number = number.split('^^')[0]
    if rel2dr[relation]['range'] in SUFFIX_MAP:
        if rel2dr[relation]['range'] == 'type.datetime':
            number = '"' + number + '-08:00' + '"'
        else:
            number = '"' + number + '"'
        number = f"{number}{SUFFIX_MAP[rel2dr[relation]['range']]}"
    else:
        print(f"{rel2dr[relation]['range']} doesn't in SUFFIX_MAP.")
    return number

def parse_pro(p):
    return_name = p.split(' = ')[0].strip()
    func_name = p.split(' = ')[1].strip().split('(')[0].strip()
    func_params = p.split(' = ')[1].strip().split('(')[1].strip().split(')')[0].strip().split(', ')
    func_params = [fp.strip(' .') for fp in func_params]
    return (return_name, func_name, func_params)

def find_lc(s1, s2): 
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

def get_normalize_number(number, sql):
    results = execute_sql_obtain_number_type(sql)
    partial_match = []
    max_l = 0
    for (dtype, nor_number) in results:
        if 'XMLSchema#float' in dtype or 'XMLSchema#int' in dtype: # number
            if float(number) == float(nor_number):
                l = 100000000
            else:
                l = 0
        elif 'XMLSchema' in dtype: # date
            date_split = nor_number.split('-')
            count_nor = 0
            for d in date_split:
                if ':' in d:
                    break
                else:
                    count_nor += 1
            count = len(number.split('-'))
            if count == count_nor and number in nor_number:
                l = 100000000
            else:
                l = 0
        elif 'literal' in dtype:
            l, cs = find_lc(number, nor_number)
        else:
            raise NotImplementedError("New dtype!")
        partial_match.append((l, dtype, nor_number))
        max_l = max(max_l, l)
    def takeFirst(elem):
        return elem[0]
    partial_match.sort(key=takeFirst, reverse=True)

    binded_number = partial_match[0]
    l, dtype, nor_number = binded_number
    if 'literal' not in dtype:
        nor_number = f'"{nor_number}"^^<{dtype}>'
    else:
        nor_number = f'"{nor_number}"'
    return nor_number

def replace_normalize_number(sql, line, value, normalization, nested_flag=False):
    if len(normalization) == 0:
        new_line = ''
    elif '^^' in normalization:
        new_line = line.replace(f' {value} ', f' {normalization} ')
    else:
        var = [s.strip() for s in line.split() if s.startswith('?var_')]
        assert len(var) == 1, line
        var = var[0]
        n_var = f"str({var})"
        new_line = line.replace(var, n_var).replace(f' {value} ', f' {normalization} ')

    if nested_flag:
        new_line = new_line.replace('var', 'nvar')
    sql = sql.replace(line, new_line)
    return sql

def convert(eid_list, ans_cla, programs):
    var_map = defaultdict()
    # programs = programs.split(';\n')
    sql_lines = []
    type_const_sql_lines = []
    obtain_number_sql_lines = []
    super_const_sql_lines = []

    has_type_constraint = True
    if ans_cla == 'common.topic':
        has_type_constraint = False
    number_value = ''
    number_line = ''
    number_var = ''
    nested_flag = False
    nested_operator = ''
    nested_var = ''
    nested_var_type = ''
    all_vars = set()
    
    start_forward_const = False
    const_var = ''
    start_backward_const = False

    for p in programs:
        if 'obtain_kg_information' in p:
            continue
        elif '>>> start constraint' in p:
            start_forward_const = True
            const_var = p.strip().split( )[-1]
            continue
        elif '<<< end constraint' in p:
            start_forward_const = False
            start_backward_const = False
            continue
        elif 'get_head_entity(' in p or 'get_tail_entity(' in p:
            cmd = parse_pro(p)
            return_name, func_name, func_params = cmd

            if start_forward_const:
                if start_backward_const:
                    var_map[return_name] = const_var
                    continue
            
            all_vars.add(return_name)
            assert len(func_params) == 2
            ent, rel = func_params
            ent = var_map[ent] if ent in var_map else ent
            all_vars.add(ent)
            if func_name == 'get_head_entity':
                line = f"?{return_name} :{rel} :{ent} ." if ent.startswith('m.') or ent.startswith('g.') else f"?{return_name} :{rel} ?{ent} ."
            elif func_name == 'get_tail_entity':
                line = f":{ent} :{rel} ?{return_name} ." if ent.startswith('m.') or ent.startswith('g.') else f"?{ent} :{rel} ?{return_name} ."
            else:
                raise NotImplementedError(f"Parse error in {p}")
            sql_lines.append(line)
            type_const_sql_lines.append(line)
            obtain_number_sql_lines.append(line)
            super_const_sql_lines.append(line.replace('var_', 'nvar_'))
        elif 'intersect(' in p:
            cmd = parse_pro(p)
            return_name, func_name, func_params = cmd
            assert len(func_params) >= 2, p
            var_map[return_name] = func_params[0]

            # modify existing node ids
            for i in range(1, len(func_params)):
                var = func_params[i]
                sql_lines = [line.replace(var, func_params[0]) for line in sql_lines]
                type_const_sql_lines = [line.replace(var, func_params[0]) for line in type_const_sql_lines]
                obtain_number_sql_lines = [line.replace(var, func_params[0]) for line in obtain_number_sql_lines]
        elif 'constraint(' in p:
            cmd = parse_pro(p)
            return_name, func_name, func_params = cmd
            if 'argmin' in func_params or 'argmax' in func_params:
                assert len(func_params) == 3
                ent, rel, operator = func_params
                ent = var_map[ent] if ent in var_map else ent
                
                line = f"?{ent} :{rel} ?{return_name} ."
                sql_lines.append(line)
                type_const_sql_lines.append(line)
                super_const_sql_lines.append(line.replace('var_', 'nvar_'))

                nested_flag = True
                # nested_operator = 'MAX' if 'max' in operator else 'MIN'
                nested_operator = 'DESC' if 'max' in operator else ''
                nested_var = return_name
                nested_var_type = get_range_type(rel)

                var_map[return_name] = ent

                if start_forward_const:
                    start_backward_const = True
            else:
                assert len(func_params) >= 4
                if len(func_params) > 4:
                    ent, rel, operator, value = func_params[0], func_params[1], func_params[2], ', '.join(func_params[3:])
                else:
                   ent, rel, operator, value = func_params
                ent = var_map[ent] if ent in var_map else ent

                if "##" in value:
                    # print(value)
                    value, rel_type = value.split('##')[0], value.split('##')[1]
                else:
                    rel_type = get_range_type(rel.replace('_to', ''))
    
                if '^^' in value:
                    value = value.split('^^')[0]
                value = value.strip(' "()')
                if 'time' in rel_type or 'date' in rel_type:
                    if operator == '=':
                        if len(value) == 'NOW' or len(value) == 4:
                            if 'from_to' in rel:
                                from_rel = rel.replace('_to', '')
                                to_rel = rel.replace('from_', '')
                            elif 'from_date' in rel:
                                from_rel = rel
                                to_rel = rel.replace('from_', 'to_')
                            else:
                                from_rel = rel
                                to_rel = rel
                            # 直接使用时间限制
                            if value == 'NOW':
                                from_date = "2015-08-10"
                                to_date = "2015-08-10"
                            elif len(value) == 4:
                                from_date = f"{value}-12-31"
                                to_date = f"{value}-01-01"
                            line = '''FILTER(NOT EXISTS {?y :{from_rel} ?sk0} || 
                                    EXISTS {?y :{from_rel} ?sk1 . 
                                    FILTER(xsd:datetime(?sk1) <= "{from_date}"^^xsd:dateTime) })
                                    FILTER(NOT EXISTS {?y :{to_rel} ?sk2} || 
                                    EXISTS {?y :{to_rel} ?sk3 . 
                                    FILTER(xsd:datetime(?sk3) >= "{to_date}"^^xsd:dateTime) })'''
                            line = line.replace('{from_rel}', from_rel).replace('{to_rel}', to_rel).replace('{from_date}', from_date).replace('{to_date}', to_date).replace('?y', f"?{ent}")
                            sql_lines.append(line)
                            type_const_sql_lines.append(line)
                            super_const_sql_lines.append(line.replace('var_', 'nvar_'))
                        else:
                            rel = rel.replace('from_to','from')
                            line = f"?{ent} :{rel} ?{return_name} ."
                            sql_lines.append(line)
                            type_const_sql_lines.append(line)
                            obtain_number_sql_lines.append(line)
                            line = '''FILTER (?y {operator} "{time}"^^xsd:dateTime)'''
                            line = line.replace('{time}', value).replace('?y', f"?{return_name}").replace('{operator}', operator)
                            sql_lines.append(line)
                            type_const_sql_lines.append(line)
                    else:
                        rel = rel.replace('from_to','from')
                        line = f"?{ent} :{rel} ?{return_name} ."
                        sql_lines.append(line)
                        type_const_sql_lines.append(line)
                        obtain_number_sql_lines.append(line)
                        line = '''FILTER (?y {operator} "{time}"^^xsd:dateTime)'''
                        line = line.replace('{time}', value).replace('?y', f"?{return_name}").replace('{operator}', operator)
                        sql_lines.append(line)
                        type_const_sql_lines.append(line)
                elif 'object.type' in rel or 'prominent_type' in rel:
                    line = f"?{ent} :{rel} :{value} ."
                    sql_lines.append(line)
                    type_const_sql_lines.append(line)
                    obtain_number_sql_lines.append(line)
                elif 'str' in rel_type or 'text' in rel_type:
                    value = value.replace('@en', '').strip('"')
                    line = f"?{ent} :{rel} ?{return_name} ."
                    sql_lines.append(line)
                    type_const_sql_lines.append(line)
                    
                    line = f'FILTER ( str(?{return_name}) {operator} "{value}" )'
                    sql_lines.append(line)
                    type_const_sql_lines.append(line)
                elif 'int' in rel_type:
                    line = f"?{ent} :{rel} ?{return_name} ."
                    sql_lines.append(line)
                    type_const_sql_lines.append(line)
                    
                    line = f"FILTER ( xsd:integer(?{return_name}) {operator} {value} )"
                    sql_lines.append(line)
                    type_const_sql_lines.append(line)
                elif 'float' in rel_type:
                    line = f"?{ent} :{rel} ?{return_name} ."
                    sql_lines.append(line)
                    type_const_sql_lines.append(line)
                    
                    line = f"FILTER ( xsd:float(?{return_name}) {operator} {value} )"
                    sql_lines.append(line)
                    type_const_sql_lines.append(line)
                elif 'enumeration' in rel_type:
                    line = f"?{ent} :{rel} ?{return_name} ."
                    sql_lines.append(line)
                    type_const_sql_lines.append(line)
                    
                    try:
                        if '.' not in value:
                            line = f"FILTER ( xsd:int(?{return_name}) {operator} {int(value)} )"
                        else:
                            line = f"FILTER ( xsd:float(?{return_name}) {operator} {float(value)} )"
                    except Exception:
                        line = f'FILTER ( str(?{return_name}) {operator} "{value}" )'
                    sql_lines.append(line)
                    type_const_sql_lines.append(line)
                else:
                    number_var = return_name

                    line = f"?{ent} :{rel} ?{return_name} ."
                    sql_lines.append(line)
                    type_const_sql_lines.append(line)
                    obtain_number_sql_lines.append(line)
                    
                    line = f'FILTER ( ?{return_name} {operator} "{value}" )'
                    number_line = line
                    sql_lines.append(line)
                    type_const_sql_lines.append(line)
                
                var_map[return_name] = ent

                if start_forward_const:
                    start_backward_const = True
        elif 'end(' in p:
            cmd = parse_pro(p)
            return_name, func_name, func_params = cmd
            assert len(func_params) == 1
            ent = func_params[0]
            ent = var_map[ent] if ent in var_map else ent
            all_vars.add(ent)
            
            # filter with answer type
            if has_type_constraint:
                line = f"?{ent} :{TYPE_OBJECT_REL} :{ans_cla} ."
                type_const_sql_lines.append(line)

            # filter not same with start mids
            if len(eid_list) > 0:
                line = f"FILTER ("
                for mid in eid_list:
                    line += f" ?{ent} != :{mid} &&"
                line = line.strip(' &')
                line += " )"
                sql_lines.append(line)
                type_const_sql_lines.append(line)
                super_const_sql_lines.append(line.replace('var_', 'nvar_'))

            # add first distinct candidates
            line = f"SELECT DISTINCT ?{ent} WHERE " + "{"
            sql_lines = [line] + sql_lines
            sql_lines.append('}')
            type_const_sql_lines = [line] + type_const_sql_lines
            type_const_sql_lines.append('}')

            # add superlative constraint
            if nested_flag:
                if 'enumeration' in nested_var_type:
                    nested_var_type = 'integer'
                if len(nested_operator) > 0:
                    if len(nested_var_type) > 0:
                        constraint_line = f"ORDER BY {nested_operator}(xsd:{nested_var_type}(?{nested_var}))\nLIMIT 1"
                    else:
                        constraint_line = f"ORDER BY {nested_operator}(?{nested_var})\nLIMIT 1"
                else:
                    if len(nested_var_type) > 0:
                        constraint_line = f"ORDER BY xsd:{nested_var_type}(?{nested_var})\nLIMIT 1"
                    else:
                        constraint_line = f"ORDER BY ?{nested_var}\nLIMIT 1"
                sql_lines.append(constraint_line)
                type_const_sql_lines.append(constraint_line)

            # add final distinct candidates
            line = f"SELECT (?{ent} AS ?value) WHERE " + "{"
            sql_lines = [line] + sql_lines
            sql_lines.append('}')
            type_const_sql_lines = [line] + type_const_sql_lines
            type_const_sql_lines.append('}')

            if len(number_value) > 0 and len(number_var) > 0:
                # add first distinct candidates for number
                line = f"SELECT DISTINCT ?{number_var} WHERE " + "{"
                obtain_number_sql_lines = [line] + obtain_number_sql_lines
                obtain_number_sql_lines.append('}')

                # add final distinct candidates for number
                line = f"SELECT (?{number_var} AS ?value) WHERE " + "{"
                obtain_number_sql_lines = [line] + obtain_number_sql_lines
                obtain_number_sql_lines.append('}')
        # const_sql = convert_constraint(eid_list, ans_cla, programs)
    final_sql = PREFIX + '\n'.join(sql_lines)
    final_sql_w_const = PREFIX + '\n'.join(type_const_sql_lines)
    if len(number_value) > 0 and len(number_var) > 0:
        obtain_number_sql = PREFIX + '\n'.join(obtain_number_sql_lines)
    else:
        obtain_number_sql = ''
    if nested_flag:
        nested_sql = PREFIX + '\n'.join(super_const_sql_lines)
    else:
        nested_sql = ''
    return final_sql, final_sql_w_const, nested_sql, (obtain_number_sql, number_value, number_line)

def unify_out_degree(programs, rid2reverse):
    programs = programs.split(';\n')
    new_programs = []
    for pro in programs:
        if 'get_head_entity' in pro:
            func = pro.split('=')[1].split('(')[0].strip()
            assert func == 'get_head_entity'
            rel = pro.split('(')[1].split(')')[0].split(', ')[1].strip()
            if rel in rid2reverse:
                rev_rel = rid2reverse[rel]
                rev_func = 'get_tail_entity' if func == 'get_head_entity' else 'get_head_entity'
                rev_pro = pro.replace(func, rev_func).replace(rel, rev_rel)
                pro = rev_pro
        new_programs.append(pro)
    return ';\n'.join(new_programs)

if __name__ == '__main__':
    data_path = sys.argv[1]
    
    # split = 'train'
    # data_name = 'webqsp'
    # with open(f'data/grailqa/{split}_w_pro_v1.json', 'r') as f:
    #     all_data = json.load(f)
    #     print(f"load {len(all_data)} samples.")
    # with open(f'data/webqsp/{split}_w_pro_v1.json', 'r') as f:
    #     all_data = json.load(f)
    #     print(f"load {len(all_data)} samples.")
    # with open(f'data/{data_name}/{split}_w_pro_v1.json', 'r') as f:
    #     all_data = json.load(f)
    #     print(f"load {len(all_data)} samples.")
    # with open(f'data/cwq/{split}_w_pro_v1_w_convert.json', 'r') as f:
    #     all_data = json.load(f)
    #     print(f"load {len(all_data)} samples.")
    if data_path.endswith('json'):
        with open(data_path, 'r') as f:
            all_data = json.load(f)
            print(f"load {len(all_data)} samples.")
    else:
        with open(data_path, 'r') as f:
            all_data = f.readlines()
            all_data = [json.loads(d) for d in all_data]
            print(f"load {len(all_data)} samples.")

    with open(f'data/par_rid_reverse_map.json','r') as f:
        rid2reverse = json.load(f)
        print(len(rid2reverse))

    total_samples = 0
    all_f1 = []
    new_all_f1 = []
    bad_case_id = []
    start = 0
    for idx, d in enumerate(tqdm(all_data[start:], total=len(all_data)-start, ncols=100)):
        total_samples += 1
        qid = d['qid']
        # if qid not in ['WebQTrn-1577_cbcb6d987a82c89fb2d8355ee01c84e8']:
        #     continue
        question = d['question']
        anno_ans = [a['answer_argument'] for a in d['answer']]
        gold_sql = d['sparql_query']
        if '#MANUAL' in gold_sql:
            continue
        # if len(anno_ans) == 0:
        #     continue
        programs = d['ground_pros'] if 'ground_pros' in d else d['programs']
        # print(programs)
        if 'var_-1' in programs:
            continue
        # new_programs = unify_out_degree(programs, rid2reverse)
        # programs = new_programs
        
        eids, ans_cla = d['start_ent_mids'], d['class_type']

        def predict_answer(qid, eids, programs):
            pred_sql, pred_sql_w_type_const, nested_sql, (obtain_number_sql, number_value, number_line) = convert(eids, ans_cla, programs)
            
            if len(obtain_number_sql) > 0:
                # try:
                number_normalize = get_normalize_number(number_value, obtain_number_sql)
                # except Exception:
                #     bad_case_id.append(qid)
                #     return [], ''
                pred_sql = replace_normalize_number(pred_sql, number_line, number_value, number_normalize)
                pred_sql_w_type_const = replace_normalize_number(pred_sql_w_type_const, number_line, number_value, number_normalize)
                # nested_sql = replace_normalize_number(nested_sql, number_line, number_value, number_normalize, True)
            
            all_ans = []
            try:
                pred_ans = execute_sql(pred_sql)
            except Exception:
                pred_ans = []
                # bad_case_id.append(qid)
            
            try:
                pred_ans_w_type_const = execute_sql(pred_sql_w_type_const)
            except Exception:
                pred_ans_w_type_const = []
                # bad_case_id.append(qid)

            return pred_ans, pred_sql, pred_ans_w_type_const, pred_sql_w_type_const
        
        try:
            pred_ans, pred_sql, pred_ans_w_type, pred_sql_w_type_const = predict_answer(qid, eids, programs.split(';\n'))
        except Exception:
            print(qid)
            print("Programs:")
            print(programs)
            # break
            bad_case_id.append(qid)
            continue
        
        final_ans = pred_ans_w_type if len(pred_ans) >= len(pred_ans_w_type) > 0 else pred_ans
        
        gold_ans = execute_sql(gold_sql)
        if len(gold_ans) == 0:
            # print(f"{qid}: Gold answer is empty!")
            continue
        
        f1 = compute_f1(gold_ans, final_ans)
        all_f1.append(f1)
        if f1 == 0:
            print('----------')
            print(qid)
            print(programs)
            print("Pred SQL:")
            print(pred_sql)
            print("Results: ", pred_ans)
            print("Gold SQL:")
            print(gold_sql)
            print("Results: ", gold_ans)
            # print("Pred SQL w Type:")
            # print(pred_sql_w_type_const)
            # print("Results: ", pred_ans_w_type)
            # break
            bad_case_id.append(qid)
        
    print(f"{round(np.mean(all_f1)*100, 2)} - {len(bad_case_id)}/{total_samples}")