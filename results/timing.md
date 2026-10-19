# Single-process timing

Machine: Intel64 Family 6 Model 151 Stepping 5, GenuineIntel | Windows-11-10.0.26200-SP0 | Python 3.12.4. Budget 30000 evaluations, one thread, mean over the repetitions [s]. eval = objective evaluations, repair = repair incl. its constraint checks, other = the algorithm itself.

## ridge-hi/none

| method | total | eval | repair | other | total / NSGA-II |
|---|---|---|---|---|---|
| MT-R | 7.8 | 4.6 | 0.0 | 3.1 | 0.88 |
| MT | 7.6 | 4.5 | 0.0 | 3.1 | 0.86 |
| FINE | 8.8 | 4.8 | 0.0 | 4.0 | 1.00 |
| NSGA3 | 10.8 | 5.0 | 0.0 | 5.8 | 1.23 |
| MOEAD | 24.8 | 8.2 | 0.0 | 16.7 | 2.83 |
| SPSO | 7.8 | 3.8 | 0.0 | 4.0 | 0.89 |
| CCMO | 10.4 | 3.9 | 0.0 | 6.6 | 1.19 |
| EMCMO | 17.2 | 3.9 | 0.0 | 13.3 | 1.96 |

## maze6/loose

| method | total | eval | repair | other | total / NSGA-II |
|---|---|---|---|---|---|
| MT-R | 14.2 | 4.7 | 5.6 | 3.9 | 1.60 |
| MT | 8.7 | 4.8 | 0.0 | 3.9 | 0.98 |
| FINE | 8.9 | 4.9 | 0.0 | 3.9 | 1.00 |
| NSGA3 | 10.7 | 5.2 | 0.0 | 5.5 | 1.20 |
| MOEAD | 25.1 | 8.6 | 0.0 | 16.5 | 2.82 |
| SPSO | 8.0 | 4.1 | 0.0 | 3.9 | 0.90 |
| CCMO | 10.2 | 4.1 | 0.0 | 6.1 | 1.14 |
| EMCMO | 10.9 | 4.1 | 0.0 | 6.9 | 1.23 |

## dem-zhemin/tight

| method | total | eval | repair | other | total / NSGA-II |
|---|---|---|---|---|---|
| MT-R | 15.5 | 3.8 | 8.1 | 3.5 | 1.88 |
| MT | 7.6 | 3.9 | 0.0 | 3.6 | 0.91 |
| FINE | 8.3 | 4.3 | 0.0 | 3.9 | 1.00 |
| NSGA3 | 10.0 | 4.6 | 0.0 | 5.5 | 1.21 |
| MOEAD | 24.6 | 8.3 | 0.0 | 16.3 | 2.97 |
| SPSO | 7.8 | 3.8 | 0.0 | 4.0 | 0.95 |
| CCMO | 10.6 | 3.8 | 0.0 | 6.8 | 1.28 |
| EMCMO | 17.1 | 3.8 | 0.0 | 13.3 | 2.06 |

## dem-qinling/tight

| method | total | eval | repair | other | total / NSGA-II |
|---|---|---|---|---|---|
| MT-R | 24.8 | 3.5 | 18.3 | 3.0 | 3.10 |
| MT | 6.6 | 3.6 | 0.0 | 3.0 | 0.82 |
| FINE | 8.0 | 4.1 | 0.0 | 3.9 | 1.00 |
| NSGA3 | 9.9 | 4.2 | 0.0 | 5.6 | 1.23 |
| MOEAD | 28.3 | 9.5 | 0.0 | 18.8 | 3.54 |
| SPSO | 7.9 | 3.9 | 0.0 | 4.0 | 0.99 |
| CCMO | 10.9 | 4.0 | 0.0 | 6.9 | 1.36 |
| EMCMO | 15.6 | 4.1 | 0.0 | 11.6 | 1.96 |

