# generated with mq config
config = {
    'scheduler': 'lsf',
    'parallel_python': 'gpaw python',
    'nodes': [
        ('XeonE5_2660v3', {'cores': 20, 'memory': '126GB'}), 
        ('XeonE5_2650v4', {'cores': 24, 'memory': '252GB'})],
    'extra_args': ['-q', 'hpc']}
