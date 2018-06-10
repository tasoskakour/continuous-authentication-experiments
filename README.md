# Continuous Authentication Experiments

## How to use

All the experiments are called from main.py file.

### Load data

Firstly, you must specify from where to load the datasets. You can either load them from database (which it automatically will be saved to a json file for later use) or you can load them from a local file

* If you want to load them from mongodb: python main.py db (ask me for mlab db credentials)
* If you want to load them from local:   python main.py local

### Functions Experiments

The experiments can be set inside the main.py file, instructions of use are specified within each function

* For pure visualization of data: visualize.my_scatter(...)

* To experiment with pure classification: classification.experiment(...)

* To experiment with classification OvO & Majority Vote: classification_maj_vote.experiment(...)

* To experiment with pure anomaly/novelty detection: anomaly.experiment(...)
