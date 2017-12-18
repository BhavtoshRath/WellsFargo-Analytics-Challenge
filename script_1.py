import re
import numpy as np


'''Assigning numeric values to the 21 distinct categories in Des2 column'''
'''Create a dictionary representation det_cat as :
[key: Des2
value: A unique number]'''

det_cat = dict()
count = 0
with open('data/daily-creditcard___detailCategory.csv') as infile:
    for line in infile:
        det_cat[line.rstrip()] = count
        count += 1


'''The following section reads masked_id and Des2 columns of 
<Daily use of a WF credit card> file'''''
'''Create a dictionary representation usage_dict as :
[key: masked_id
value: temporal sequence of Des2 for the user]'''

usage_dict = dict()
with open('data/Daily_use_of_a_WF_credit_card.csv') as infile:
    for line in infile:
        l_spl = re.split('\t', line.rstrip())
        if int(l_spl[0]) in usage_dict.keys():
            usage_dict[int(l_spl[0])].append(det_cat[l_spl[3].rstrip()])
        else:
            usage_dict[int(l_spl[0])] = [det_cat[l_spl[3].rstrip()]]

user_features = {}

for masked_id in usage_dict:
    user_features[masked_id] = np.asarray(usage_dict[masked_id], dtype='int')






















