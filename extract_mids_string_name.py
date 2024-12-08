import json
import os
from collections import defaultdict
import re
import pickle
import argparse

# Get entity names from FastRDFStore
# https://github.com/microsoft/FastRDFStore

from struct import *


class BinaryStream:
    def __init__(self, base_stream):
        self.base_stream = base_stream

    def readByte(self):
        return self.base_stream.read(1)

    def readBytes(self, length):
        return self.base_stream.read(length)

    def readChar(self):
        return self.unpack('b')

    def readUChar(self):
        return self.unpack('B')

    def readBool(self):
        return self.unpack('?')

    def readInt16(self):
        return self.unpack('h', 2)

    def readUInt16(self):
        return self.unpack('H', 2)

    def readInt32(self):
        return self.unpack('i', 4)

    def readUInt32(self):
        return self.unpack('I', 4)

    def readInt64(self):
        return self.unpack('q', 8)

    def readUInt64(self):
        return self.unpack('Q', 8)

    def readFloat(self):
        return self.unpack('f', 4)

    def readDouble(self):
        return self.unpack('d', 8)

    def decode_from_7bit(self):
        """
        Decode 7-bit encoded int from str data
        """
        result = 0
        index = 0
        while True:
            byte_value = self.readUChar()
            result |= (byte_value & 0x7f) << (7 * index)
            if byte_value & 0x80 == 0:
                break
            index += 1
        return result

    def readString(self):
        length = self.decode_from_7bit()
        return self.unpack(str(length) + 's', length)

    def writeBytes(self, value):
        self.base_stream.write(value)

    def writeChar(self, value):
        self.pack('c', value)

    def writeUChar(self, value):
        self.pack('C', value)

    def writeBool(self, value):
        self.pack('?', value)

    def writeInt16(self, value):
        self.pack('h', value)

    def writeUInt16(self, value):
        self.pack('H', value)

    def writeInt32(self, value):
        self.pack('i', value)

    def writeUInt32(self, value):
        self.pack('I', value)

    def writeInt64(self, value):
        self.pack('q', value)

    def writeUInt64(self, value):
        self.pack('Q', value)

    def writeFloat(self, value):
        self.pack('f', value)

    def writeDouble(self, value):
        self.pack('d', value)

    def writeString(self, value):
        length = len(value)
        self.writeUInt16(length)
        self.pack(str(length) + 's', value)

    def pack(self, fmt, data):
        return self.writeBytes(pack(fmt, data))

    def unpack(self, fmt, length = 1):
        return unpack(fmt, self.readBytes(length))[0]


def get_key(subject):
    if subject.startswith("m.") or subject.startswith("g."):
        if len(subject) > 3:
            return subject[0:4]
        elif len(subject) > 2:
            return subject[0:3]
        else:
            return subject[0:2]
    else:
        if len(subject) > 1:
            return subject[0:2]
        return subject[0:1]

def is_cvt(subject):
    tp_key = get_key(subject)
    if tp_key in cvt_nodes:
        if subject in cvt_nodes[tp_key]:
            return cvt_nodes[tp_key][subject]
    return False

def is_ent(ent_str):
    str_pattern = re.compile(r"^[mg]\..+$")
    if type(ent_str) is not bool and str_pattern.match(ent_str):
        return True
    else:
        return False


def obtain_all_mids(data_path):
    all_mids = set()
    with open(data_path, 'r') as f:
        all_data = json.load(f)

    for d in all_data:
        s_exp = d['s_expression']
        if s_exp is None:
            continue
        eles = s_exp.split()
        for e in eles:
            e = e.strip().replace(')', '')
            if is_ent(e):
                all_mids.add(e)
            # else:
            #     if 'm.' in e or 'g.' in e:
            #         print(e)
    return all_mids

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--mid_mapping_path', required=True)
    parser.add_argument('--output_path', required=True)

    args = parser.parse_args()

    print("Start extracting all mids string name.")
    return args


if __name__ == '__main__':
    args = _parse_args()
    input_path = args.input_path
    output_path = args.output_path
    mid_mapping_path = args.mid_mapping_path

    all_mids = set()
    if 'webqsp' in input_path:
        for split in ['train', 'test']:
            path = input_path.replace("SPLIT", split)
            mids = obtain_all_mids(path)
            all_mids.update(mids)
        print("Total %d mids" % len(all_mids))

    all_mids = list(all_mids)

    # load mapping
    ALL_ENTITY_NAME_BIN = os.path.join(mid_mapping_path, "FastRDFStore_data", "namesTable.bin")
    entity_names = {}
    with open(ALL_ENTITY_NAME_BIN, 'rb') as inf:
        stream = BinaryStream(inf)
        dict_cnt = stream.readInt32()
        print("total entities:", dict_cnt)
        for _ in range(dict_cnt):
            key = stream.readString().decode()
            # if key.startswith('m.') or key.startswith('g.'):
            # key = '/' + key[0] + '/' + key[2:]
            # if not key.startswith('m.') and not key.startswith('g.'):
            #     print("Key: %s"%(key))
            value = stream.readString().decode()
            entity_names[key] = value

    ALL_CVT_NAME_BIN = os.path.join(mid_mapping_path, "FastRDFStore_data", "cvtnodes.bin")
    with open(ALL_CVT_NAME_BIN, 'rb') as cvtf:
        reader = BinaryStream(cvtf)
        dictionariesCount = reader.readInt32()
        print("total cvt entities:", dictionariesCount)
        cvt_nodes = {}
        for i in range(0, dictionariesCount):
            key = bytes.decode(reader.readString())
            # covert byte to string
            count = reader.readInt32()
            # print(key, count)
            dict_tp = {}
            for j in range(0, count):
                mid = bytes.decode(reader.readString())
                isCVT = reader.readBool()
                dict_tp[mid] = isCVT
            cvt_nodes[key] = dict_tp

    mid_subset = defaultdict()
    cvt_subset = defaultdict()
    no_mapping_subset = set()
    for mid in all_mids:
        cvt_subset[mid] = is_cvt(mid)
        if mid in entity_names:
            mid_subset[mid] = entity_names[mid]
        elif mid:
            no_mapping_subset.add(mid)

    print("There are %d--%d entities mapping--no mapping" % (len(mid_subset), len(no_mapping_subset)))
    data = (cvt_subset, mid_subset)
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
        print("Save the mapping dict to %s" % output_path)

