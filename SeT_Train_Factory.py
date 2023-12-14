"""
Contains batch training for Seismic Transformer traning (SeT_Train.py).

Author: Jason Jiang (Xunfun Lee)
Date: 2023.12.3
"""

import subprocess

# define the params combinition:
# num_layer = num_head = 12 is better for 100k traning data, bigger model won't increase the performance
# learning_rate=0.001(bigger loss will become nan); weight_decay=0.0(bigger acc will decrease); lr_decay=0.1/0.2(very similar)
params_list = [
    # {'patch_size': 250, 'hidden_size': 768, 'num_layer': 12, 'num_head': 12, 'batch_size': 1972, 'epoch': 20, 'learning_rate': 0.001, 'weight_decay': 0.0, 'mlp_dropout': 0.1, 'plot': False},        # lr decay = 0.2
]

# run each of the params
for params in params_list:
    # build comman line
    cmd = [
        'python', 'SeT_Train.py',
        '--patch_size', str(params['patch_size']),
        '--hidden_size', str(params['hidden_size']),
        '--num_layer', str(params['num_layer']),
        '--num_head', str(params['num_head']),
        '--batch_size', str(params['batch_size']),
        '--epoch', str(params['epoch']),
        '--learning_rate', str(params['learning_rate']),
        '--weight_decay', str(params['weight_decay']),
        '--mlp_dropout', str(params['mlp_dropout']),        
        '--plot', str(params['mlp_dropout']),
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # print the process bar
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    process.poll()