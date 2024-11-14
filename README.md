# Dynamical systems for craving

Substance Use Disorders (SUD) can be modeled as a prospective link from cues to craving and use.
To explore the nonlinear interplay between craving and cues, this study applied dynamical systems theory (DST) to ecological momentary assessment (EMA) data.
Optimized linear Seasonal Auto-Regressive Integrated Moving Average with eXogenous variable (SARIMAX) models were used to phenotype patients with a SUD (alcohol, tobacco, cannabis, opiates, and cocaine), considering there is potentially a complex interaction between cues exposure reports, and craving intensity in daily life.
These phenotypic profiles were reproduced in computational DST models to analyze nonlinear interactions between cues, craving, and use.
This study involved 211 individuals and 8,260 observations; 154 patients fitted with the SARIMAX model of the influence of cues on craving, and 57 patients fitted with the SARIMAX model of a possible influence of craving on cues.
Two DST models were adjusted to reproduce the complex temporal dynamics of SUD according to these two respective directions of influence.
The first DST model (adjusted to the influence of cues on craving) showed that an increase in environmental cues led to a rise in craving, which then diminished both the cues and craving itself, with use patterns following craving’s lead.
This profile of patients was driven by a phenomenon of “maximum cue saturation.”
The second DST model (adjusted to the influence of craving on cues) revealed that an increase in craving increased perception of cues, which in turn increased craving and led to use, with use peaking and then reducing craving.
This profile of patients is driven by a phenomenon of “maximum use saturation.”
These models both highlight craving as a crucial mediator between cues and use and open new therapeutic avenues.


## Project Structure

- **src/** : contains the main Python files for SARIMAX and DST analyses.
- **.npy** : folders to store simulated data.
- **requirements.txt** : list of dependencies needed to run the project.
- **README.md** : this file – providing an overview of the project and usage instructions.


## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ChristopheGauld/Dynamical_systems_for_craving.git
