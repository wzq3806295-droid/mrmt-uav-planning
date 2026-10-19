# Extra statistics for the results section

## Pairwise tests on final HV (A vs B; W/T/L for A, Holm over instances)

| A vs B | dem-qinling/none | dem-qinling/loose | dem-qinling/tight | dem-longmen/none | dem-longmen/loose | dem-longmen/tight | dem-zhemin/none | dem-zhemin/loose | dem-zhemin/tight | W/T/L |
|---|---|---|---|---|---|---|---|---|---|---|
| MT-R vs FINE | > (0.001) | > (0.001) | = (1.000) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | > (0.048) | 8/1/0 |
| MT-R vs NSGA3 | > (0.001) | > (0.001) | = (1.000) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | = (0.749) | 7/2/0 |
| MT-R vs MOEAD | > (0.001) | > (0.001) | = (1.000) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | > (0.004) | 8/1/0 |
| MT-R vs SPSO | > (0.001) | > (0.001) | = (1.000) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | > (0.015) | 8/1/0 |
| MT-R vs CCMO | > (0.001) | > (0.001) | = (1.000) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | = (0.545) | 7/2/0 |
| MT-R vs EMCMO | > (0.001) | > (0.001) | = (1.000) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | < (0.042) | 7/1/1 |
| MT vs FINE | > (0.002) | > (0.002) | = (1.000) | > (0.002) | > (0.015) | > (0.008) | > (0.002) | > (0.002) | = (0.071) | 7/2/0 |
| MT-R vs MT | = (1.000) | = (1.000) | = (1.000) | = (1.000) | = (1.000) | = (1.000) | = (1.000) | = (1.000) | = (1.000) | 0/9/0 |

## Pairwise tests on HV at 25 % budget (A vs B; W/T/L for A, Holm over instances)

| A vs B | dem-qinling/none | dem-qinling/loose | dem-qinling/tight | dem-longmen/none | dem-longmen/loose | dem-longmen/tight | dem-zhemin/none | dem-zhemin/loose | dem-zhemin/tight | W/T/L |
|---|---|---|---|---|---|---|---|---|---|---|
| MT-R vs FINE | > (0.001) | > (0.001) | = (1.000) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | = (0.126) | 7/2/0 |
| MT-R vs NSGA3 | > (0.001) | > (0.001) | = (1.000) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | = (0.620) | 7/2/0 |
| MT-R vs MOEAD | > (0.001) | > (0.001) | = (1.000) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | > (0.027) | 8/1/0 |
| MT-R vs SPSO | > (0.001) | > (0.001) | = (1.000) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | > (0.013) | 8/1/0 |
| MT-R vs CCMO | > (0.001) | > (0.001) | = (1.000) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | = (1.000) | 7/2/0 |
| MT-R vs EMCMO | > (0.001) | > (0.001) | = (1.000) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | > (0.001) | = (1.000) | 7/2/0 |
| MT vs FINE | > (0.001) | > (0.001) | = (1.000) | > (0.001) | > (0.015) | > (0.008) | > (0.001) | > (0.001) | > (0.020) | 8/1/0 |
| MT-R vs MT | = (1.000) | = (1.000) | = (1.000) | = (1.000) | = (1.000) | = (1.000) | = (1.000) | = (1.000) | = (1.000) | 0/9/0 |

## Evaluations until the first feasible fine solution (median; 'never' = not within budget)

| instance | FINE | NSGA3 | MOEAD | SPSO | CCMO | EMCMO | MT | MT-R |
|---|---|---|---|---|---|---|---|---|
| dem-qinling/none | 2000 | 2000 | 2040 | 8000 | 2000 | 2000 | 2000 | 2000 |
| dem-qinling/loose | 6000 | 4000 | never | 7000 | 4000 | 4000 | 2000 | 2000 |
| dem-qinling/tight | never | never | never | never | never | never | never | never |
| dem-longmen/none | 2000 | 2000 | 18060 | 5000 | 2000 | 2000 | 2000 | 2000 |
| dem-longmen/loose | never | never | never | 4000 | never | never | 2000 | 2000 |
| dem-longmen/tight | never | never | never | 2000 | never | never | 2000 | 2000 |
| dem-zhemin/none | 2000 | 2000 | 2040 | 3000 | 2000 | 2000 | 2000 | 2000 |
| dem-zhemin/loose | 2000 | 2000 | 18000 | 2000 | 2000 | 2000 | 2000 | 2000 |
| dem-zhemin/tight | never | 17000 | never | 11000 | 3000 | 4000 | 2000 | 4000 |

(instances where every variant is feasible at the first record are omitted)

## Runs ending with at least one feasible solution (out of 10)

| instance | FINE | NSGA3 | MOEAD | SPSO | CCMO | EMCMO | MT | MT-R |
|---|---|---|---|---|---|---|---|---|
| dem-qinling/none | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| dem-qinling/loose | 6 | 7 | 1 | 9 | 10 | 10 | 10 | 10 |
| dem-qinling/tight | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dem-longmen/none | 10 | 10 | 7 | 9 | 10 | 10 | 10 | 10 |
| dem-longmen/loose | 1 | 0 | 1 | 10 | 0 | 0 | 7 | 10 |
| dem-longmen/tight | 1 | 0 | 0 | 10 | 0 | 0 | 9 | 10 |
| dem-zhemin/none | 10 | 10 | 8 | 10 | 10 | 10 | 10 | 10 |
| dem-zhemin/loose | 10 | 10 | 5 | 10 | 10 | 10 | 10 | 10 |
| dem-zhemin/tight | 3 | 5 | 1 | 9 | 9 | 10 | 8 | 8 |

## Repair constraint checks per run, as a fraction of the evaluation budget (median)

| instance | MT-R |
|---|---|
| dem-qinling/none | 0.003 |
| dem-qinling/loose | 0.015 |
| dem-qinling/tight | 0.102 |
| dem-longmen/none | 0.002 |
| dem-longmen/loose | 0.010 |
| dem-longmen/tight | 0.005 |
| dem-zhemin/none | 0.002 |
| dem-zhemin/loose | 0.010 |
| dem-zhemin/tight | 0.046 |
