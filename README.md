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

## GENIE 12.1

* https://www.synapse.org/#!Synapse:syn32309524

## GENIE 13.1

* https://www.synapse.org/#!Synapse:syn51355584

# Downloading from Synapse

Download datasets using their synids.

```python
from nextgenlp.synapse import sync_datasets
synids = ["syn51355584"]  # GENIE v13.1 dataset
files = sync_datasets(synids)
```

By default, the `sync_datasets` function will download synapse datasets
to the `DATA_PATH/synapse` directory specified in the `config.ini` file
(if they are not already there).
It will also return a list of `synapseclient.entity.File` objects with
metadata about the files that were just synced.


# Getting Started

Create a GENIE 13.1 dataset object

```python
from nextgenlp.genie import GenieData

# you will have to update this path
syn_base_path = "/path/to/syn51355584"

# set read_cna=True to read copy number alteration data 
gd = GenieData.from_synapse_directory(syn_base_path, read_cna=False)
```
