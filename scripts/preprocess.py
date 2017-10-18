# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json
import os
import six
import numpy as np
import h5py
import codecs


input_txt = '../../../datasets/SocialMedia/captions_1M_alltxt.txt'
output_h5 = '../../../datasets/SocialMedia/captions_1M_alltxt.h5'
output_json ='../../../datasets/SocialMedia/captions_1M_alltxt.json'
val_frac = 0.1
test_frac = 0.1
quiet = False
encoding = 'utf-8'


if __name__ == '__main__':
  if encoding == 'bytes': encoding = None

  # First go the file once to see how big it is and to build the vocab
  token_to_idx = {}
  total_size = 0
  with codecs.open(input_txt, 'r', encoding) as f:
    for line in f:
      total_size += len(line)
      for char in line:
        if char not in token_to_idx:
          token_to_idx[char] = len(token_to_idx) + 1

  # Now we can figure out the split sizes
  val_size = int(val_frac * total_size)
  test_size = int(test_frac * total_size)
  train_size = total_size - val_size - test_size

  if not quiet:
    print('Total vocabulary size: %d' % len(token_to_idx))
    print('Total tokens in file: %d' % total_size)
    print('  Training size: %d' % train_size)
    print('  Val size: %d' % val_size)
    print('  Test size: %d' % test_size)

  # Choose the datatype based on the vocabulary size
  dtype = np.uint8
  if len(token_to_idx) > 255:
    dtype = np.uint32
  if not quiet:
    print('Using dtype ', dtype)

  # Just load data into memory ... we'll have to do something more clever
  # for huge datasets but this should be fine for now
  train = np.zeros(train_size, dtype=dtype)
  val = np.zeros(val_size, dtype=dtype)
  test = np.zeros(test_size, dtype=dtype)
  splits = [train, val, test]

  # Go through the file again and write data to numpy arrays
  split_idx, cur_idx = 0, 0
  with codecs.open(input_txt, 'r', encoding) as f:
    for line in f:
      for char in line:
        splits[split_idx][cur_idx] = token_to_idx[char]
        cur_idx += 1
        if cur_idx == splits[split_idx].size:
          split_idx += 1
          cur_idx = 0

  # Write data to HDF5 file
  with h5py.File(output_h5, 'w') as f:
    f.create_dataset('train', data=train)
    f.create_dataset('val', data=val)
    f.create_dataset('test', data=test)

  # For 'bytes' encoding, replace non-ascii characters so the json dump
  # doesn't crash
  if encoding is None:
    new_token_to_idx = {}
    for token, idx in six.iteritems(token_to_idx):
      if ord(token) > 127:
        new_token_to_idx['[%d]' % ord(token)] = idx
      else:
        new_token_to_idx[token] = idx
    token_to_idx = new_token_to_idx

  # Dump a JSON file for the vocab
  json_data = {
    'token_to_idx': token_to_idx,
    'idx_to_token': {v: k for k, v in six.iteritems(token_to_idx)},
  }
  with open(output_json, 'w') as f:
    json.dump(json_data, f)
