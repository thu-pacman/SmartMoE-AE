# SmartMoE AE

This is the artifact repository for paper #15 at ATC'23, titled **SmartMoE: Efficiently Training Sparsely-Activated Models through Combining Static and Dynamic Parallelization**.

The repository includes:

- Log files in `logs/` and scripts in `plotting/`. These were used for plotting the figures in the SmartMoE paper.
- Source code in `src/`. Both SmartMoE and baselines codes are provided.
- Trace data in `moe_trace/`. An expert selection dataset was collected for evaluations.
- Scripts in `scripts/`. Out-of-the-box scripts are provided for running  experiments.

It should be noticed that reproducing the original SmartMoE experiments requires strict hardware requirements: 32 NVIDIA A100 GPUs and 64 NVIDIA V100 GPUs are necessary. To overcome hardware limitations, we have made available the raw data of SmartMoE experiments and scripts for reproducing experiments on 16 NVIDIA V100 GPUs.

## Getting Started

### Quick Reproduction: Plotting from Raw Data (~2minutes)

**Hardware requirements: No GPUs are needed.**

**Software requirements: Python and some commonly used packages (e.g. Matplotlib and NumPy).**

The Only command needed to plot all figures in the evaluation section is: 

```
./RUNME-a.sh
```

This may takes a few minutes to complete. Once you have successfully run this command, you will find a directory named `outputs_from_log_${TIME_YOU_RUN}`  which contains the generated figures `fig[8-13].pdf`.

In detail, the `RUNMME-a.sh` will read original log files, perform some post-processing, and plot the figures. The generated figures will be exactly the same as those in SmartMoE's submission version of the paper.

### In-depth Reproduction: Plotting from Real Run(~2hours)

**Hardware requirements: 16 NVIDIA V100 GPUs.**

**Software requirements: Distributed training software (e.g. CUDA, NCCL and PyTorch) and plotting software (e.g. Matplotlib and NumPy).**

First, please follow the [guideline](#installation) to set up the environment.

Once your environment is ready, simply run the following command to reproduce the experiments:

```
./RUNME-b.sh
```

This may takes several hours to complete. Once you have successfully run this command, you will find a directory named `outputs_from_exec_${TIME_YOU_RUN}`  which contains the generated figures. As reproductions were performed on a smaller cluster ( 16 NVIDIA V100 GPUs),  the generated figures may be different from those in the paper. We discuss differences in experiments results [here](#discussion).

**Note**: Please note that the provided scripts have only been tested on a cluster managed by the [Slurm](https://www.schedmd.com/) scheduler and [Spack](https://github.com/spack/spack/) package manager. If your cluster uses different management software, modifications to the scripts may be necessary.

## Installation

### SmartMoE

SmartMoE is implemented based on [FastMoE](https://github.com/laekov/fastmoe). It requires CUDA, NCCL, and PyTorch (v1.10.0). To install PyTorch, please use the following command:

```  bash
pip install --user -f https://download.pytorch.org/whl/cu113/torch_stable.html torch==1.10.0+cu113
```

In addition, the developer package of NCCL (≥2.9.9) is required, and its version should be the same as PyTorch (e.g. PyTorch v1.10.0 installed via the above command comes with NCCL 2.10.3). You can be download it from https://developer.nvidia.com/nccl/nccllegacy-downloads.

After that, compile and install SmartMoE with the following command in `./src/fastmoe`:

```bash
# PWD: ./src/fastmoe
USE_NCCL=1 python setup.py install --user
```

If different types of GPUs are installed in the node for compilation and test runs, you should provide the environment variable `TORCH_CUDA_ARCH_LIST`. Please refer to https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension for more information.

### Megatron-LM

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) was used as the GPT training framework in SmartMoE's experiments, and NVIDIA Apex is required. However, the apex package on PyPI is broken, so you need to install it from source. Clone it from https://github.com/NVIDIA/apex, and install it using the following command:

```
python3 setup.py install --user --cuda_ext --cpp_ext
```

### DeepSpeed

[DeepSpeed](https://github.com/microsoft/Deepspeed) `v0.6.1` was used in DeepSpeed's experiments. To install it,  use the following command:

```
pip install --user deepspeed==0.6.1
```

We compare SmartMoE with an optimized MoE implementation based on DeepSpeed. For a fair comparison, we choose a DeepSpeed version that supports 3D parallelism like Megatron-LM, and it can be found at https://github.com/microsoft/Megatron-DeepSpeed/tree/moe (we copy it into `src/Megatron-DeepSpeed`).

## Discussion

You can reproduce SmartMoE's experiments on a smaller cluster where only 16 NVIDIA V100 GPUs are required by running `./RUNME-b.sh`. Please note that the generated figures may differ from those in the paper due to hardware differences.

### Section 6.3: End-to-End Speedup

In the paper, sec.6.3 has figures 8 and 9 as its results. However, in the reproduction, **only the first group bars (labeled as `16/1.2` on inky) in figure 8 are plotted**.

We decide not to reproduce all data for two reasons. Firstly, full experiments require tens of hours, which is  time-consuming. Secondly, performing the reproduction on a cluster with 64 NVIDIA V100 GPUs and 32 NVIDIA A100 GPUs can be expensive.

### Section 6.4: Performance of Skeleton

In the paper, sec.6.4 has figure 10 and 11 as its results. However, in the reproduction, **only the second group bars (labeled as `16/+inf` ) in figure 10 are plotted**.

Due to time limitations, we have decided not to reproduce the other data in figures 10 and 11. Nonetheless, the selected portion is  representative enough to explain the effectiveness of SmartMoE's  algorithm.

### Section 6.5: Effect of Adaptive Parallelization

In the paper, sec.6.5 has figure 12, 13 as its results; **both two figures are reproduced**.

It is important to note that the reproduced figures may not be exactly identical to the original ones due to differences in hardware and model structure.

