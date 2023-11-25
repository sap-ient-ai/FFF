# Summary of FFF
    FFF shows promise to exponentially reduce the compute-power required by a feed-forward neural network layer

    I (π) have provided my best interpretation of this innovation in [doc/theory.md](doc/theory.md)

# Goal
The purpose of this repo is to play with FastFeedForward Networks.

We plan to engineer a performant implementation.

Also as the innovation is new, we want to explore & tweak, rather than simply dive 100% into optimizations.

Chat with us on [Discord](https://discord.gg/Utv7tcGz)

NOTE: We're not affiliated with the authors of the FFF papers, but we'd be thrilled if they were to pop in and say hi!

# Papers
    - (18 Sep 2023)[Fast Feedforward Networks](https://arxiv.org/pdf/2308.14711.pdf) ([code](https://github.com/pbelcak/fastfeedforward))

    - (15 Nov 2023)[Exponentially Faster Language Modelling](https://arxiv.org/abs/2311.10770) ([code](https://github.com/pbelcak/FastBERT/benchmark_pytorch/fff/fff_bmm.py))

    Second revision of paper has updated repo [here](https://github.com/pbelcak/UltraFastBERT) containing CUDA code.

# Misc

2023.11.23
    π created https://github.com/pbelcak/UltraFastBERT/issues/1
        Observing the BERT benchmark performs slower than the vanilla BERT on HF
    
    π created https://github.com/pbelcak/UltraFastBERT/issues/2
        An interpretation of the core algorithm, and a suggestion for improvement (remove the .gelu)
            Links to a gist demo of FFF operating over MNIST.
