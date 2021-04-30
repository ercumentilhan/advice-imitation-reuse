# Advice Imitation & Reuse

This repository contains the source code for the advice collection, imitation and reuse algorithm proposed in the paper titled ["Learning on a Budget via Teacher Imitation"](https://arxiv.org/abs/2104.08440) which is an extended version of the algorithm presented in ["Action Advising with Advice Imitation in Deep Reinforcement Learning"](https://arxiv.org/abs/2104.08441) ([codes](https://github.com/ercumentilhan/naive-advice-imitation)).

## Requirements

**Packages (and their tested versions with Python 3.7):**
- numpy=1.19.4
- gym=0.18.0
- tensorflow=2.2.0
- pathlib2=2.3.5

**Environment:**
- [OpenAI Gym - Arcade Learning Environment](https://github.com/openai/gym/blob/master/docs/environments.md#atari)

## Execution

The code can be executed simply by running
```
python code/main.py
```

The input arguments with their default values can be found in `code/main.py`.

---

The teacher agent(s) to be used can be set in `code/constants/general.py` with their saved model checkpoints as follows:
```python
TEACHER = {
    'ALE-Enduro': ('ALE24V0_EG_000_20201105-130625', '0', 6000e3),
    'ALE-Freeway': ('ALE26V0_EG_000_20201105-172634', '0', 3000e3),
    'ALE-Pong': ('ALE43V0_EG_000_20201106-011948', '0', 5800e3),
    'ALE-Qbert': ('ALE46V0_EG_000_20201023-120616', '0', 7000e3),
    'ALE-Seaquest': ('ALE50V0_EG_000_20201019-132350', '0', 7000e3),
}
```
where each dictionary entry uses the following format:
```
<Game Name>: (<Model directory>, <Model subdirectory (seed)>, <Checkpoint (timesteps)>)
```
We provide these example teacher models in the `checkpoints` directory.

---

While the desired run configurations can be executed by providing the relevant arguments and their values as follows,

```
python code/main.py --run-id ALE26V0_NA --process-index 0 --machine-name HOME --n-training-frames 5000000 --env-name ALE-Enduro
```

these can also be defined in `code/constants/ale.py`  with configuration IDs to be managed and run easily as an alternative. For instance, a predefined option with ID 1000 can be run in Pong with experimnt seed 0 as follows:

```
python code/main.py --env-name ALE-Pong --config-set 1000 --seed 0
```

These are the configuration sets that are already defined in the file (ID: Short description, as further detailed in the paper):

- **1000:** NA (No advising, training from scratch)
- **2000:** EA (Early advising)
- **2100:** RA (Random advising)
- **5000:** AR (Advice reuse)
- **6000:** AR+A (AR with automatic threshold tuning)
- **7000:** AR+A+E (AR+A with unrestricted reuse procedure)
- **8000:** AIR (Advice Imitation & Reuse, AR+A+E with the uncertainty-based advice collection strategy)
