# LiteRate
LiteRate is a program implementing birth-death models within a Bayesian framework to estimates diversification dynamics from cultural occurence data. The model detects statistically-significant rate shifts in the history of the cultural population that theoretically correspond with major historical events and/or the action of evolutionary processes. 

<figure align="center">
<img src="https://github.com/dsilvestro/LiteRate/raw/master/other/Figure_3.png" alt="" width="500" height="700" border="0">
</figure>

# Restricted Models
The repository also contains some more restricted birth-death models for testing hypotheses generated by LiteRate. For now, these include DDRate (rates driven by diversity-dependent competition) and TrendRate (rates driven by exogenous trends).

# Cite
To learn more about methods or if you use these models in published work, please cite both of:

Gjesfjeld, Erik, Daniele Silvestro, Jonathan Chang, Bernard Koch, Jacob G. Foster, and Michael E. Alfaro. ‘A Quantitative Workflow for Modeling Diversification in Material Culture’. PLOS ONE 15, no. 2 (6 February 2020): e0227579. https://doi.org/10.1371/journal.pone.0227579.

Koch, Bernard, Daniele Silvestro, and Jacob G. Foster. n.d. “The Evolutionary Dynamics of Cultural Change (as Told Through the Birth and Brutal, Blackened Death of Metal Music).” SocArXiv. [https://doi.org/10.31235/osf.io/659bt10.31235/osf.io/659bt](https://doi.org/10.31235/osf.io/659bt10.31235/osf.io/659bt).

# Tutorials
Detailed tutorials (ipython notebooks) demonstrating the use of the software package will soon be available [here](http://www.dysoc.org/cesmodules/). In the meantime, you can read draft PDFs of each tutorial in the repository:

- [Introduction](https://github.com/dsilvestro/LiteRate/raw/master/other/0_Introduction_Tutorial.pdf)
- [Diversity and Diversification](https://github.com/dsilvestro/LiteRate/raw/master/other/1_Diversity%20and%20Diversification.pdf)
- [Introduction to LiteRate](https://github.com/dsilvestro/LiteRate/raw/master/other/2_Introduction_to_LiteRate.pdf)
- [Interpreting LiteRate Results](https://github.com/dsilvestro/LiteRate/raw/master/other/3_Interpreting_LiteRate_Results.pdf)
- [Modeling Evolutionary Mechanisms](https://github.com/dsilvestro/LiteRate/raw/master/other/4_Modeling_Evolutionary__Mechanisms_in_Diversification_Rates.pdf)
- [Cultural Phylogenies](https://github.com/dsilvestro/LiteRate/raw/master/other/S1_Cultural_Phylogenies.pdf) (supplemental)


### Requirements
LiteRate is written in Python v.3 and requires 
the libraries numpy and scipy. 
Source files and installers are available [here](https://numpy.org) and [here](https://scipy.org). 

Alternatively, you can download the requirements.txt file and use `pip install -r requirements.txt` to install them from the terminal (UNIX systems). Note that if Python v.2 is the default version of Python in your machine you might need to use `python3` to launch the program and `pip3` to install the required libraries. 



