import json
from tqdm import tqdm

def process_one_rel(lines):
    rel_dict = {}
    relation = lines[0][0]
    property = False
    class_type = False
    for l in lines:
        h, r, t = l[0].strip(), l[1].strip(), l[2].strip()
        if r == 'type.object.type':
            if t == 'type.property':
                property = True
            elif t == 'type.type':
                class_type = True
            else:
                return {}
        elif r in ['type.object.name', 'rdf-schema#domain', 'rdf-schema#range', 'type.property.reverse_property']:
            if '#' in r:
                rel_dict[r.split('#')[-1]] = t
            else:
                rel_dict[r.split('.')[-1]] = t
    if property:
        rel_dict['rid'] = relation
        return rel_dict
    elif class_type:
        if relation.startswith('g.') or 'name' not in rel_dict:
            return {}
        rel_dict['cid'] = relation
        return rel_dict
    else:
        return {}

with open('KB/FastRDFStore_FB/fb_en_nonM.txt', 'r') as f:
    lines = []
    inval_count = 0
    pre_rel = ''
    all_rid_info = []
    all_cid_info = []
    count = 0
    for line in tqdm(f, total=472208746, ncols=100):
        count += 1
        if count % 1000000 == 0:
            print(all_rid_info[-1])
            print(all_cid_info[-1])
            print(f"Process {count} lines, where {inval_count} lines and {len(all_rid_info)} relations information and {len(all_cid_info)}.")
        eles = line.strip('\n').split('\t')
        if len(eles) < 3:
            inval_count += 1
            continue
        cur_rel = eles[0]
        if cur_rel != pre_rel:
            pre_rel = cur_rel
            if len(lines) > 0:
                rid_info = process_one_rel(lines)
                if len(rid_info) > 0:
                    if 'rid' in rid_info:
                        all_rid_info.append(rid_info)
                    elif 'cid' in rid_info:
                        all_cid_info.append(rid_info)
                lines = [eles]
            else:
                lines.append(eles)
        else:
            lines.append(eles)
with open('data/rid_to_info.json', 'w') as f:
    json.dump(all_rid_info, f)
with open('data/cid_to_info.json', 'w') as f:
    json.dump(all_cid_info, f)