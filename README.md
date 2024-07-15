# ASGCN

## Clone
If you'd like to clone this project, 
just run the following command. 
To include sub-projects, 
simply add the `--recurse-submodules` command line parameter.
```shell
git clone https://github.com/032004129xuzhiyong/ASGCN.git --recurse-submodules
```

## Dependencies
We apply several specific python libraries, install them if you want to run.
```shell
bash env.sh
```

## Overview
Here, we provide an implementation of ASGCN in pytorch. 
The structure of the repository is organized as follows:
* `config/`: contains the configuration files for the model.
* `data/`: contains the original data but not includes `animals` dataset because of its large size.
* `datasets/`: contains the code for loading and processing the data.
* `models/`: contains the implementation of the model.
* `mytool`: contains the tool scripts.
* `env.sh`: contains the environment download script.
* `run.py`: contains the main function for the project.

## Usage
In order to reproduce the presented performance, 
it is necessary to execute the following command from the command line. 
The configuration file should be selected 
according to the specific dataset in question.
```shell
python run.py run -cps config/YaleB_F10.yaml -q
```
At the end of the run, the `best/` directory will appear, 
and the results of the run are stored in the `conf.yaml` file (e.g., `best/YaleB_F10/conf.yaml`).