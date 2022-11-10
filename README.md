# Hack4NF 2022

* https://hack4nf.bemyapp.com/

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
Generate one by folling the instructions at the links below, 

* https://help.synapse.org/docs/Client-Configuration.1985446156.html
* https://www.synapse.org/#!PersonalAccessTokens

Next, copy `secrets.json.template` to `DATA_PATH/secrets.json` 
and edit the new file to contain your personal access token. 


# Access to GENIE

## GENIE 12.0

* https://www.synapse.org/#!Synapse:syn32309524

## GENIE 13.3 (special access request required)

* https://www.synapse.org/#!Synapse:syn36709873

Start by requesting access to GENIE dataset 

* https://www.synapse.org/#!Synapse:syn34548529/wiki/618904

After you get a confirmation email - you will need to go back the the site, click "Request Access" and click "Accept" on the electronic terms that pop up. You should then have permission to download the dataset. 

# Downloading from Synapse

Download datasets using their synids. 
 
```python
from nextgenlp.synapse import sync_datasets
synids = ["syn32309524"]  # GENIE v12.0 dataset
files = sync_datasets(synids)
```

By default, the `sync_datasets` function will download synapse datasets 
to the `DATA_PATH/synapse` directory specified in the `config.ini` file
(if they are not already there).
It will also return a list of `synapseclient.entity.File` objects with 
metadata about the files that were just synced.


# Getting Started 

Create a GENIE dataset object with ... 





