# POET

This repo contains implementation of the POET algorithm described in:

[Paired Open-Ended Trailblazer (POET): Endlessly Generating Increasingly Complex and Diverse Learning Environments and Their Solutions](https://arxiv.org/abs/1901.01753)

An article on Uber Engineering Blog describing POET can be found [here](https://eng.uber.com/poet-open-ended-deep-learning/).

## Requirements

- [Fiber](https://uber.github.io/fiber/)
- [OpenAI Gym](https://github.com/openai/gym)

## Run the code locally

To run locally on a multicore machine

```./run_poet_local.sh final_test```

## Run the code on a computer cluster

To containerize and run the code on a computer cluster (e.g., Google Kubernetes Engine on Google Cloud), please refer to [Fiber Documentation](https://uber.github.io/fiber/getting-started/#containerize-your-program).
