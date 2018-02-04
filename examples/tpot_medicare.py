# TO RUN
# python experiments/run_experiment.py examples/tpot_medicare.py <PEM FILE> <BUCKET> --instance-type m5.4xlarge

from tpot import TPOTClassifier

local = False
random_state = 42
features_file = 'data/2015_partB_sparse.npz'
labels_file = 'data/2015_partB_lookup.csv'
label_col = 'provider_type'

classifier = TPOTClassifier(generations=5, population_size=5, cv=5,
                            random_state=random_state, verbosity=2, scoring='accuracy',
                            max_time_mins=None, max_eval_time_mins=5)
