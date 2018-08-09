import argparse
import lmdb
import random
import sys

parser = argparse.ArgumentParser(description='LMDB splitter')
parser.add_argument('--input',help='input lmdb')
parser.add_argument('--test-output',help='output lmdb (for test)')
parser.add_argument('--val-output',help='output lmdb (for validation)')
parser.add_argument('--test-prop', help='test proportion (val proportion is 1-test_prop)',
                    default= .5, type=float)
parser.add_argument('--test-num', help='number of items in test db', type=int)
args = parser.parse_args()



db = lmdb.open(args.input,readonly=True)
txn = db.begin()
cursor = txn.cursor()
data = []
for key, value in cursor:
    data.append([key, value])

all_indexes = set(xrange(len(data)))
if args.test_num is not None:
    test_indexes = random.sample(all_indexes,int(args.test_num))
else:
    test_indexes = random.sample(all_indexes,int(float(len(data))*args.test_prop))
val_indexes = list(all_indexes - set(test_indexes))


map_size = (sys.getsizeof(data[0][0]) + sys.getsizeof(data[0][1])) * len(data)

test_db = lmdb.open(args.test_output, map_size = map_size)
with test_db.begin(write=True) as txn:
    for i in test_indexes:
        txn.put(data[i][0], data[i][1])

val_db = lmdb.open(args.val_output, map_size=map_size)
with val_db.begin(write=True) as txn:
    for i in val_indexes:
        txn.put(data[i][0], data[i][1])


