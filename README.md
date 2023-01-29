# Introduction

This repo is a place to continue development on ideas that came from participation in [Hack4NF 2022](https://hack4nf-platform.bemyapp.com).

Code developed during the hackathon will remain archived [here](https://github.com/MocoMakers/hack4nf-2022).


# Install `nextgenlp` Python Package

Create and activate a python environment with your favorite tool.
An example for conda would be,

```bash
conda create --name ng310 python=3.10
conda activate ng310
```

Run the following command in the directory that contains the `setup.cfg` file.
You might have to update to the latest version of pip

```bash
pip install -U pip
```

```bash
pip install -e .
```


# Setup Data Config

## config.ini

Copy `config.ini.template` to `config.ini` and edit the line that starts with `DATA_PATH`.
This should point to an empty directory.
`nextgenlp` will use this location to store synapse datasets and derived data.


## secrets.json

In order to securely download data from synapse you will need a personal access token.
Generate one by follwing the instructions at the links below,

* https://help.synapse.org/docs/Client-Configuration.1985446156.html
* https://www.synapse.org/#!PersonalAccessTokens

Next, copy `secrets.json.template` to the `SECRETS_PATH` specified in the `config.ini` file and rename it to `secrets.json`.
By default `SECRETS_PATH` = `DATA_PATH/secrets.json` but you can change
this to whatever you want. Finally, add your personal access token to the `secrets.json` file.


# Access to GENIE

## GENIE 12

* https://www.synapse.org/#!Synapse:syn32309524

## GENIE 13

* link when public

# Downloading from Synapse

Download datasets using their synids.

```python
from nextgenlp.synapse import sync_datasets
synids = ["syn32309524"]  # GENIE v12 dataset
files = sync_datasets(synids)
```

By default, the `sync_datasets` function will download synapse datasets
to the `DATA_PATH/synapse` directory specified in the `config.ini` file
(if they are not already there).
It will also return a list of `synapseclient.entity.File` objects with
metadata about the files that were just synced.


# Getting Started

Create a GENIE 12 dataset object

```python
# you will have to update this
syn_base_path = "/path/to/syn32309524"

gene_panels = os.path.join(syn_base_path, "gene_panels")
data_clinical_patient = os.path.join(syn_base_path, "data_clinical_patient.txt")
data_clinical_sample = os.path.join(syn_base_path, "data_clinical_sample.txt")
data_mutations_extended = os.path.join(syn_base_path, "data_mutations_extended.txt")
data_CNA = os.path.join(syn_base_path, "data_CNA.txt")
gd = genie.GenieData.from_file_paths(
    gene_panels,
    data_clinical_patient,
    data_clinical_sample,
    data_mutations_extended,
    data_CNA,
)
```
