
## Audio Network Hacking

---

## Overview

### Models


1) Image VAE


    |                   | Data type | Output distribution | Reconstruction loss  |
    |-------------------|:---------:|--------------------:|---------------------:|
    | BMNIST VAE        | Binary    | Bernoulli           | Binary cross-entropy |
    | MNIST VAE         | Greyscale | Normal              | Mean-squared error   |
    | MNIST $\beta$-VAE | Greyscale | Normal              | Mean-squared error   |

2) Audio VAE 

    __to do__

## Code

### Paths

- Config paths: `code/config/models/{model_type}_config.json`
- Log directory: `saved_models/log/{model_name}/`
- Checkpoint directory: `saved_models/models/{model_name}/`
### Set-up

1) Create a Conda virtual environment if necessary:

        conda env create -f ./[env_for_your_os.yml]

    This will create a new Conda environment named `env_anh`.
    It should have include all required packages for everything to run.

2) Activate the environment:

    a) In VScode:

    - Run (Ctrl+Shift+P) the `Python: Select Interpreter` command.
    - Select the python path matching `env_anh`.
    You may need to refresh the interpreter list 

![vscoder_interpreter](./doc/resources/images/vscode_interpreter.png)

    b) from the command line

        conda activate env_anh
### Training

a) In VScode:

- Select the `Train: [Your Model]` option in the 'Run and Debug' tab.
- Open a checkpoint in the checkpoint dir and select 
the `Resume: [Your Model]` option in the 'Run and Debug' tab.


![vscoder_interpreter](./doc/resources/images/vscode_training.png)

b) From the command line:

```bash
    python3 ./code/train.py --config "your_path_to_model_config_file.json" [--resume "your_path_to_model_checkpoint_file.pkl"])
```

#### Using TensorBoard

a) In VScode:

    - Run (Ctrl+Shift+P) the `Python: Launch TensorBoard` command.
    - Select 'Use current working directory'.
    - The TensorBoard watcher should open in a new view. 

![vscoder_interpreter](./doc/resources/images/vscode_tensorboard.png)


b) From the command line:

```bash
    tensorboard --logdir "your_log_dir"
```

- Then follow the given address in a web browser
(ex: http://localhost:6006).
- Refer to the [Models](###Models) section for information
on logging directories for each model type.

### Testing

a) In VScode:
- Open a checkpoint in the checkpoint directory
- Select the `Test: [Your Model]` option in the 'Run and Debug' tab.



b) From the command line:

```bash
    python3 ./code/test.py --config "your_path_to_model_config_file.json" --resume "your_path_to_model_checkpoint_file.pkl")
```

- Refer to the [Models](###Models) section for information
on configuration and checkpoint files for each model type.

## Data set

All data sets are automatically downloaded if not found in toy/.
For images:
- MNIST
For audio:
- VSO2


## References

- [Subject](./doc/atiam2021_Chemla.pdf)
