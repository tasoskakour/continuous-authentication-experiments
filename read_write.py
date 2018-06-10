# pylint: disable = C0111, C0103, C0411, C0301, W0102
"""This module is used to read data from the external mongodb database,
   or to read data from the local drive (e.g. from json)
   or to write data to loca drive (e.g. to json).
   \n Mainly this module will be used as a library."""
import pymongo
import json
import bson.json_util
import numpy as np
from copy import deepcopy

# Ask me for DATABASE credentials if you want :)
DATABASE_URI = '<mongodb uri>'


def load_from_mongodb(query={}):
    """Loads data from the external mongodb database.
    """
    print '\nLoading docs from external database.\n'
    client = pymongo.MongoClient(DATABASE_URI)
    db = client['keystroke-dynamics-database']
    collection = db['keystrokedatamodels']
    if query == {}:
        cur = collection.find({})
    return cur


def write_docs_to_local(docs_cursor, filename='subjects-docs'):
    """Writes raw docs as loaded from mongo to local"""
    print 'Writing subjects docs to local drive.\n'
    with open(filename + '.json', 'w') as fout:
        json.dump(json.loads(bson.json_util.dumps(deepcopy(docs_cursor))), fout)


def read_docs_from_local(filename='subjects-docs'):
    """Reads raw subjects docs to local drive"""
    print '\n Reading subjects docs from local drive.'
    with open(filename + '.json', 'r') as fin:
        return json.loads(fin.read())


def write_timings_to_local(data, filename='subjects-data', convert_np_to_list=True):
    """Writes subjects-data as json to local drive."""
    print '\nWriting subjects timings data to local drive.\n'
    tmp = deepcopy(data)
    with open(filename + '.json', 'w') as fout:
        # Convert numpy arrays to list of lists :/
        for s in tmp:
            for p in s['data']:
                p['points'] = p['points'].tolist()
        json.dump(tmp, fout)


def read_timings_from_local(filename='subjects-data.json'):
    """Reads json from local drive."""
    print '\nLoading subjects timings data  from local.\n'
    with open(filename) as data_file:
        data = json.loads(data_file.read())
    # Convert list of lists back to numpy arrays :/
    tmp = deepcopy(data)
    for s in tmp:
        for p in s['data']:
            p['points'] = np.array(p['points'])
    return tmp
