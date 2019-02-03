# LiteRate
Fast estimation of birth and death rates from large data setusing reversible jump MCMC

Modified code from [PyRate](https://github.com/dsilvestro/PyRate) designed for Cultural diversification



### multi-trait extinction model (MTE)
example input file: `example_data/Example_multiDiscreteTraitDep.txt`

Run analysis:
`python MTEmodel.py -d example_data/Example_multiDiscreteTraitDep.txt -t0 23 -t1 8`
where `t0` and `t1` specify the boundaries of the time window of interest. If not specified the entire dataset will be analyzed.

To plot the results use `plot_MTE_results.R`
