# Physics-aided-DRL

EnsembleQ-test.py is the main file used to train an RL model.

Hyperparameters and functions are defined in utils.py.


>python Collect-dataset.py > cd.log 2>&1 &

>python Initialize-model.py > im.log 2>&1 &

>python Trained-model.py --model initialized-actor --func 1 > tm.log 2>&1 &

>python Trained-model.py --model EnsembleQ-actor-1-r1 --func 1 > tm.log 2>&1 &

>python Trained-model.py --model EnsembleQ-actor-1-r2 --func 1 > tm.log 2>&1 &

>python Trained-model.py --model EnsembleQ-actor-1-r3 --func 1 > tm.log 2>&1 &

>python Trained-model.py --model EnsembleQ-actor-1-r1-init-0 --func 1 > tm.log 2>&1 &

>python Trained-model.py --model EnsembleQ-actor-2 --func 2 > tm.log 2>&1 &

>python Trained-model.py --model EnsembleQ-actor-3 --func 3 > tm.log 2>&1 &

>python ND-compare.py --func 1 > ndc.log 2>&1 &

>python EnsembleQ-test.py --n 2 --m 2 --rindex 1 > n2m2.log 2>&1 &

>python EnsembleQ-test.py --n 2 --m 2 --rindex 1 --init 1 > n2m2.log 2>&1 &
