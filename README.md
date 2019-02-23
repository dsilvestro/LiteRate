# LiteRate
Fast estimation of birth and death rates from large data setusing reversible jump MCMC

Modified code from [PyRate](https://github.com/dsilvestro/PyRate) designed for Cultural diversification

# Logistic Niche Growth Model

Nested model where birth and death rates are a function of diversity and the growth of a niche through a logistic curve

# SimulateRate

Simulator for birth death rates under different processes. Currently written for logistic niche growth but adaptable. Also simple set up for approximate Bayesian computation.

### multi-trait extinction model (MTE)
example input file: `example_data/Example_multiDiscreteTraitDep.txt`

Run analysis:
`python MTEmodel.py -d example_data/Example_multiDiscreteTraitDep.txt -t0 23 -t1 8`
where `t0` and `t1` specify the boundaries of the time window of interest. If not specified the entire dataset will be analyzed.

To plot the results use `plot_MTE_results.R`
