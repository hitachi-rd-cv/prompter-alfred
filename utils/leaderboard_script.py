import pickle
import argparse
import json
from glob import glob
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dn_startswith', type=str, required=True)
parser.add_argument('--json_name', type=str)

args = parser.parse_args()

if args.json_name is None:
    args.json_name = args.dn

lens = list()
all_actions = set()
results = {'tests_unseen': [{'trial_T20190908_010002_441405': [{'action': 'LookDown_15', 'forceAction': True}]}],
           'tests_seen': [{'trial_T20190909_042500_949430': [{'action': 'LookDown_15', 'forceAction': True}]}]}
seen_strs = ['seen', 'unseen']
for seen_str in seen_strs:
    pickle_globs = glob("results/leaderboard/actseqs_test_" +
                        seen_str + "_" + args.dn_startswith + "_" + "*")

    pickles = []
    for g in pickle_globs:
        pickles += pickle.load(open(g, 'rb'))

    total_logs = []
    ep_num = list()
    for i, t in enumerate(pickles):
        key = list(t.keys())[0]
        actions = t[key]
        trial = key[1]
        total_logs.append({trial: actions})
        ep_num.append(key[0])

    for i, (t, ep_n) in enumerate(zip(total_logs, ep_num)):
        key = list(t.keys())[0]
        actions = t[key]
        new_actions = []
        for indx, action in enumerate(actions):
            if action['action'] == 'LookDown_0' or action['action'] == 'LookUp_0':
                pass
            else:
                all_actions.add(action['action'])
                new_actions.append(action)

            # if indx > 950:
            #     break
        assert len(new_actions) < 1000
        lens.append(indx)
        total_logs[i] = {key: new_actions}

    print(max(lens))

    print(len(total_logs))
    if seen_str == 'seen':
        assert len(total_logs) == 1533
        results['tests_seen'] = total_logs
    else:
        assert len(total_logs) == 1529
        results['tests_unseen'] = total_logs

print(len(results['tests_seen']))
print(len(results['tests_unseen']))
print(len(all_actions))
print(all_actions)

if not os.path.exists('leaderboard_jsons'):
    os.makedirs('leaderboard_jsons')

save_path = 'leaderboard_jsons/tests_actseqs_' + args.json_name + '.json'
with open(save_path, 'w') as r:
    json.dump(results, r, indent=4, sort_keys=True)
