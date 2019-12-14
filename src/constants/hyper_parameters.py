RELATION_AGE_DEEP_WALK_HYPER_PARAMS = [
    # Hyper-parameters 1
    ' '.join([
        '--number-walks 10',
        '--representation-size 64',
        '--seed 699807',
        '--walk-length 40',
        '--window-size 5',
        '--undirected True'
    ]),
    # Hyper-parameters 2
    ' '.join([
        '--number-walks 10',
        '--representation-size 16',
        '--seed 699807',
        '--walk-length 10',
        '--window-size 3',
        '--undirected True',
        '--vertex-freq-degree'
    ]),
    # Hyper-parameters 3
    ' '.join([
        '--number-walks 10',
        '--representation-size 32',
        '--seed 699807',
        '--walk-length 25',
        '--window-size 7',
        '--undirected True',
        '--vertex-freq-degree'
    ]),
    # Hyper-parameters 4
    ' '.join([
        '--number-walks 10',
        '--representation-size 10',
        '--seed 699807',
        '--walk-length 40',
        '--window-size 5',
        '--undirected False'
    ])
]

RELATION_AGE_NODE2VEC_HYPER_PARAMS = [
    # Hyper-parameters 1
    {
        'dimensions': 16,
        'walk_length': 20,
        'num_walks': 10,
        'p': 1,
        'q': 1
    },
    # Hyper-parameters 2
    {
        'dimensions': 16,
        'walk_length': 20,
        'num_walks': 10,
        'p': 2,
        'q': 2
    },
    # Hyper-parameters 3
    {
        'dimensions': 16,
        'walk_length': 20,
        'num_walks': 10,
        'p': 0.5,
        'q': 0.5
    },
    # Hyper-parameters 4
    {
        'dimensions': 16,
        'walk_length': 20,
        'num_walks': 10,
        'p': 0.5,
        'q': 2
    },
    # Hyper-parameters 5
    {
        'dimensions': 16,
        'walk_length': 20,
        'num_walks': 10,
        'p': 2,
        'q': 0.5
    }
]