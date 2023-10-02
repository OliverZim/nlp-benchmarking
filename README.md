# Benchmarking deep learning best practices

This project can be used to benchmark common best practices in deep learning (e.g. model-compilation and mixed-precision-training) and evaluate the actual impact of these methods on training speed. This [report](https://api.wandb.ai/links/nlp_benchmarks/p9dltkjf) serves as an example for possible measurements.

This work is based on the [NLP research template](https://github.com/konstantinjdobler/nlp-research-template).

## Setup
### Requirements
This project is specifically designed to allow for reproducable results. We utilize ```conda-lock``` to provide reproducable environments. The recommended way to install ```conda-lock``` is using ```pipx install conda-lock```.

To log the experiments we recommend using [Weights&Biases](https://wandb.ai/). Before running any experiments you have to set your WANDB_API_KEY, WANDB_ENTITY and WANDB_PROJECT as environment variables. To persist these variables you can specify them in your ```.bashrc```-file (e.g. ```export WANDB_PROJECT=deep_learning_benchmarks```).
### Basics
The recommended way to use this project is by running the experiments inside a dedicated docker container. You can either use the already prebuilt [oliverzim/nlp-benchmarks](https://hub.docker.com/r/oliverzim/nlp-benchmarks)-image or build your own image with the ```Dockerfile``` cointained in the repository. The image contains all the necessary dependencies specified by the lockfile ```conda-lock.yml```. If you want to share your environments you can do so by sharing this lockfile.
### Updating the dependenices
To change the dependencies you need to execute the following steps.
1. Specify new dependencies in the ```environment.yml``` file.
2. Update  ```conda-lock.yml``` to contain the new dependencies. -> Simply run ```conda-lock``` in the directory with your ```environment.yml``` file.
3. Rebuild the ```Dockerfile``` so it incorporates the changes. -> ```docker build --tag <username>/<imagename>:<tag> --platform=linux/amd64 .```


## Usage
This template contains 2 scripts to make it easier for you to run your experiments.
<details><summary><code>console.sh</code></summary>
<p>

The ```console.sh```-script is used to start an interactive session inside the research environment. It lets you specify which gpu devices you want to use for the experiments, allows for persistent caching (e.g. for tokenizers and preprocessed datasets) and automatically mounts the necessary directories into the docker container.
To execute this script run ```bash ./scripts/console.sh``` from the directory that contains the ```train.py``` file.

</details>
<details><summary><code>benchmarks.sh</code></summary>
<p>

In the ```benchmarks.sh```-script you specify the experiments you want to run by setting different flags for program execution. A simple example to understand the effect of mixed-precision training could look like this:
```bash
# add your benchmarks here:
python train.py --val_before_training=False --training_goal=200000 -d ./data/sw --wandb_run_name=precision-16   --batch_size_per_device=42 --workers=1 --precision=16-mixed --compile=true --force_deterministic=false
python train.py --val_before_training=False --training_goal=200000 -d ./data/sw --wandb_run_name=precision-32   --batch_size_per_device=42 --workers=1 --precision=32 --compile=true --force_deterministic=false
```

To execute this script run ```bash ./scripts/benchmarks.sh``` from the directory that contains the ```train.py``` file.

</details>
<p>

In case you want to run your experiments on a remote server you have to make sure that your connection to the server is not closed while executing your experiments. A workaround for this would be using the terminal multiplexer ```tmux```.

A typical workflow for your experiments could look like this:
1. Write code for your experiment (e.g. in ```train.py``` or ```benchmarks.py```).
2. Specify the experiments in the ```benchmarks.sh```-file.
2. Start a named terminal-session: ```tmux new -s EXPERIMENT_NAME```
3. Start the interactive research environment with the GPUs you want to use: ```bash ./scripts/console.sh```
4. Execute the experiments: ```bash ./scripts/benchmarks.sh```
5. Wait until execution (and monitor progress live with Weights&Biases).
6. Close the terminal-session: ```tmux kill-session EXPERIMENT_NAME```