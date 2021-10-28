# Physics-aided-DRL

EnsembleQ-test.py is the main file used to train an RL model.

Hyperparameters and functions are defined in utils.py.

ND-compare.py evaluates the performance of AMIX-ND algorithm.

Trained-model.py evaluates the performance of NN-based algorithms.


>python ND-compare.py --findex 1 --adjust 0 > ndc.log 2>&1 &
>python ND-compare.py --findex 2 --adjust 0 > ndc.log 2>&1 &
>python ND-compare.py --findex 3 --adjust 0 > ndc.log 2>&1 &

>python ND-compare.py --findex 1 --adjust 1 > ndc.log 2>&1 &
>python ND-compare.py --findex 2 --adjust 1 > ndc.log 2>&1 &
>python ND-compare.py --findex 3 --adjust 1 > ndc.log 2>&1 &

>python EnsembleQ-test.py --n 2 --m 2 --rindex 1 --gamma 0.99 > n2m2.log 2>&1 &
>python EnsembleQ-test.py --n 2 --m 2 --rindex 2 --gamma 0.99 > n2m2.log 2>&1 &
>python EnsembleQ-test.py --n 2 --m 2 --rindex 3 --gamma 0.99 > n2m2.log 2>&1 &
>python EnsembleQ-test.py --n 2 --m 2 --rindex 4 --gamma 0.99 > n2m2.log 2>&1 &

>python Trained-model.py --model actor-r-1-gamma-0.99 --findex 1 > tm.log 2>&1 &
>python Trained-model.py --model actor-r-2-gamma-0.99 --findex 1 > tm.log 2>&1 &
>python Trained-model.py --model actor-r-3-gamma-0.99 --findex 1 > tm.log 2>&1 &
>python Trained-model.py --model actor-r-4-gamma-0.99 --findex 1 > tm.log 2>&1 &
