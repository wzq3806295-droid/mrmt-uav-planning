# IGD+ of the final solutions

Normalisation and reference set: merged non-dominated front of all methods per instance;
infeasible run = inf. Median IGD+ per instance (lower is better); W/T/L of MT-R (Holm).

## Synthetic (12)

| instance | MRMT-R | NSGA-II | NSGA-III | MOEA/D | SPSO | CCMO | EMCMO |
|---|---|---|---|---|---|---|---|
| ridge/none | 0.0306 | 0.0804 | 0.1458 | 0.0759 | 0.3423 | 0.1276 | 0.0857 |
| ridge/loose | 0.0164 | 0.0408 | 0.0385 | 0.0554 | 0.2476 | 0.0394 | 0.0318 |
| ridge/tight | 0.0132 | 0.0374 | 0.0454 | 0.0639 | 0.5734 | 0.0502 | 0.0314 |
| ridge-hi/none | 0.0101 | 0.1280 | 0.0929 | 0.3837 | 0.2950 | 0.0832 | 0.0840 |
| ridge-hi/loose | 0.0085 | 0.0640 | 0.0568 | 0.0465 | 0.2281 | 0.0433 | 0.0426 |
| ridge-hi/tight | 0.0124 | 0.0442 | 0.0430 | 0.0417 | 0.5124 | 0.0356 | 0.0436 |
| maze4/none | 0.1105 | 0.1858 | 0.1198 | 0.0736 | 0.5923 | 0.1089 | 0.0971 |
| maze4/loose | 0.0578 | 0.0393 | 0.0243 | inf | inf | 0.0384 | 0.0374 |
| maze4/tight | 0.0408 | 0.0418 | 0.0266 | inf | inf | 0.0458 | 0.0424 |
| maze6/none | 0.0128 | 0.1101 | 0.1896 | 0.1097 | 0.5140 | 0.1103 | 0.1196 |
| maze6/loose | 0.1637 | 0.3909 | 0.2257 | inf | inf | 0.3285 | 0.4654 |
| maze6/tight | 0.2332 | 0.3830 | 0.0965 | inf | inf | 0.1734 | 0.1178 |

## Real terrain (9)

| instance | MRMT-R | NSGA-II | NSGA-III | MOEA/D | SPSO | CCMO | EMCMO |
|---|---|---|---|---|---|---|---|
| dem-qinling/none | 0.0209 | 0.4222 | 0.3575 | 0.8217 | 2.2130 | 0.1607 | 0.1637 |
| dem-qinling/loose | 0.0435 | 0.3340 | 0.1711 | inf | 2.8402 | 0.1030 | 0.1097 |
| dem-qinling/tight | inf | inf | inf | inf | inf | inf | inf |
| dem-longmen/none | 0.0527 | 0.1982 | 0.1627 | 0.5221 | 2.2534 | 0.1798 | 0.1301 |
| dem-longmen/loose | 0.0495 | inf | inf | inf | 0.6247 | inf | inf |
| dem-longmen/tight | 0.0356 | inf | inf | inf | 0.5432 | inf | inf |
| dem-zhemin/none | 0.0246 | 0.4643 | 0.2868 | 1.0009 | 0.5806 | 0.2183 | 0.1886 |
| dem-zhemin/loose | 0.0823 | 0.1514 | 0.1928 | inf | 0.8268 | 0.1333 | 0.1207 |
| dem-zhemin/tight | 0.1265 | inf | inf | inf | 2.3532 | 0.0647 | 0.0618 |

## W/T/L of MT-R on IGD+

| instances | NSGA-II | NSGA-III | MOEA/D | SPSO | CCMO | EMCMO |
|---|---|---|---|---|---|---|
| Synthetic (12) | 7/5/0 | 8/1/3 | 11/1/0 | 12/0/0 | 8/4/0 | 8/4/0 |
| Real terrain (9) | 7/2/0 | 7/2/0 | 8/1/0 | 8/1/0 | 7/2/0 | 7/1/1 |
| All (21) | 14/7/0 | 14/5/2 | 18/3/0 | 20/1/0 | 14/6/1 | 14/6/1 |
