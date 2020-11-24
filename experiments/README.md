# Experiments

In this directory we present all our experiments.

If you decide to re-run the code, you may want to delete the `~/temp/data_deepobs` directory, that will be created to share the data sets among experiments, afterwards.

We provide an overivew of all experiments here, splitting them up in experiments that were presented in the main body of our paper, ones that are shown in the appendix, and a set of experiments that were not included in the paper at all.

All experiments include some `run.py` file to reproduce the results. Our original results are also stored in each experiment folder in a `results.zip` file. The `plot.py` script produces the Figures shown in the paper.

| Experiment                 | Description                                                                                                         |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 01_benchmark               | Benchmark the run time of individual instruments and configurations of Cockpit.<br>Reproduces Figure 6-7 and 12-15. |
| 02_LINE                    | Illustrative Example of why just monitoring the loss is not enough.<br>Reproduces Figure 1.                         |
| 04_benchmark_memory        | Benchmark the memory consumption of the histograms.<br>Reproduces Figure 11                                         |
| 06_preprocessing           | Study how incorrectly scaled data is represented in Cockpit.<br>Reproduces Figure 3.                                |
| 07_learning_rate_selection | Grid search the learning rate and compare the alpha values vs. final performance.<br>Reproduces Figure 5.           |
| 09_layerwise               | Analyzing the layerwise histogram for two different architectures.<br>Reproduces Figure 4.                          |
| 10_showcase                | Showcase of the full Cockpit for DeepOBS problems.<br>Reproduces Figure 2 and 17.                                   |
| 11_histogram2d             | Benchmark the performance of different implementations for computing the 2D Histogram.<br>Reproduces Figure 16.     |
| 12_alpha_explanation       | Illustrative plot explaining the Alpha Quantity.<br>Reproduces Figure 9.                                            |
| debug                      | Additional debugging experiments.                                                                                   |

## Main Text Experiments

#### Figure 1: Loss is not Enough - [`02_LINE`](02_LINE/README.md)

![Loss is not Enough](02_LINE/output/LINE.png)

#### Figure 2: Showcase - [`10_showcase`](10_showcase/README.md)

![Showcase CIFAR-100](10_showcase/output/cifar100_allcnnc_log.png)

#### Figure 3: Misscaled Data - [`06_preprocessing`](06_preprocessing/README.md)

![Misscaled Data](06_preprocessing/output/exp06.png)

#### Figure 4: Layerwise Histogram - [`09_layerwise`](09_layerwise/README.md)

![Layerwise Histogram](09_layerwise/output/exp09.png)

#### Figure 5: Tuning the Learning Rate with Alpha - [`07_learning_rate_selection`](07_learning_rate_selection/README.md)

![Mean Alpha vs Performance](07_learning_rate_selection/output/mean_alpha_vs_performance.png)

#### Figure 6 & 7: Run Time Benchmark - [`01_benchmark`](01_benchmark/README.md)

![Run Time Benchmarks](01_benchmark/output/exp01a.png)
![Run Time Benchmarks](01_benchmark/output/exp01b.png)

## Appendix Experiments

Figure 8 is not included in this list, as it is a code example. The code for it can be found in a more extensive variant in the [examples directory](../examples/00_mnist.py).
Figure 10 is a conecptual sketch of the gradient tests without any empirical results.

#### Figure 9: Motivational Sketch for Alpha - [`12_alpha_explanation`](12_alpha_explanation/README.md)

![Alpha Explanation](12_alpha_explanation/output/alpha_explanation.png)

#### Figure 11: Memory Consumption Histogram - [`04_benchmark_memory`](04_benchmark_memory/README.md)

![Memory Benchmarks](04_benchmark_memory/output/exp04.png)

#### Figure 12 - 15: Additional Run Time Benchmarks - [`01_benchmark`](01_benchmark/README.md)

#### Figure 16: Implementation Performance 2D Histogram - [`11_histogram2d`](11_histogram2d/README.md)

![Performance Histogram](11_histogram2d/output/exp11.png)

#### Figure 17: Additional Showcases - [`10_showcase`](10_showcase/README.md)

![Showcase MNIST LogReg](10_showcase/output/mnist_logreg_log.png)
![Showcase Quadratic Deep](10_showcase/output/quadratic_deep_log.png)