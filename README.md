# POET

This repo contains implementation of the POET algorithm described in:

[Paired Open-Ended Trailblazer (POET): Endlessly Generating Increasingly Complex and Diverse Learning Environments and Their Solutions](https://arxiv.org/abs/1901.01753)

An article on Uber Engineering Blog describing POET can be found [here](https://eng.uber.com/poet-open-ended-deep-learning/).

## Requirements

- [ipyparallel](https://github.com/ipython/ipyparallel)
- [OpenAI Gym](https://github.com/openai/gym)

## Run the code locally
To run locally on a multicore machine

1) On one terminal window

```ipcluster start -n NUM_WORKERS```

2) Wait until you see:

```[IPClusterStart] Engines appear to have started successfully```

3) On another terminal window:

```./run_poet_local.sh final_test```

## Run the code over the cluster

This requires set up ipyparallel over the cluster of machines, please refer to [IpyParallel Documentation](https://ipyparallel.readthedocs.io/en/latest/).


