import os.path as osp
import numpy as np
import lmdb
import pyarrow as pa
import warnings

warnings.filterwarnings('ignore')
LMDB_DEFAULT_SIZE = 1099511627776


def create_db(filepath, readonly=False):
    db = lmdb.open(filepath,
                   map_size=LMDB_DEFAULT_SIZE,
                   readonly=readonly,
                   meminit=False,
                   map_async=True)
    return db


def open_db(filepath, readonly=True):
    db = lmdb.open(filepath,
                   subdir=osp.isdir(filepath),
                   readonly=readonly,
                   lock=False,
                   readahead=False,
                   meminit=False)
    return db


def write_txn_once(txn, k, v):
    txn.put(k.encode('ascii'), pa.serialize(v).to_buffer())


def write_txn_batch(txn, keys, values):
    assert len(keys) == len(values)
    for k, v in zip(keys, values):
        txn.put(k.encode('ascii'), pa.serialize(v).to_buffer())


def write_db_once(db, k, v):
    with db.begin(write=True) as txn:
        write_txn_once(txn, k, v)


def write_db_batch(db, keys, values):
    with db.begin(write=True) as txn:
        write_txn_batch(txn, keys, values)


def read_txn_once(txn, k):
    data = pa.deserialize(txn.get(k.encode('ascii')))
    if type(data).__module__ == np.__name__:
        return np.copy(data)
    else:
        return data


def read_txn_batch(txn, keys):
    data = [read_txn_once(txn, k) for k in keys]
    return data


def read_db_once(db, k):
    with db.begin(write=False) as txn:
        return read_txn_once(txn, k)


def read_db_batch(db, keys):
    with db.begin(write=False) as txn:
        return read_txn_batch(txn, keys)
