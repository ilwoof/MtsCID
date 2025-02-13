# MtsCID
MtsCID: Multivariate Time Series Anomaly Detection by Capturing Coarse-Grained Intra- and Inter-Variate Dependencies

## Abstract
Multivariate time series anomaly detection is essential for failure management in web application operations, as it directly influences the effectiveness and timeliness of implementing remedial or preventive measures. This task is often framed as a semi-supervised learning problem, where only normal data are available for model training, primarily due to the labor-intensive nature of data labeling and the scarcity of anomalous data. Existing semi-supervised methods often detect anomalies by capturing intra-variate temporal dependencies and/or inter-variate relationships to learn normal patterns, flagging timestamps that deviate from these patterns as anomalies. However, these approaches often fail to capture salient intra-variate temporal and inter-variate dependencies in time series due to their focus on excessively fine granularity, leading to suboptimal performance. In this study, we introduce MtsCID, a novel semi-supervised multivariate time series anomaly detection method. MtsCID employs a dual network architecture: one network operates on the attention maps of multi-scale intra-variate patches for coarse-grained temporal dependency learning, while the other works on variates to capture coarse-grained inter-variate relationships through convolution and interaction with sinusoidal prototypes. This design enhances the ability to capture the patterns from both intra-variate temporal dependencies and inter-variate relationships, resulting in improved performance. Extensive experiments across seven widely used datasets demonstrate that MtsCID achieves performance comparable or superior to state-of-the-art benchmark methods.


<p align="center">
<img src=".\figures\framework_overview.png" height = "250" alt="" align=center />
</p>

## Project Structure

```
├─data  # Data files.
├─data_factory      # data preprocessing, data loader, etc.
├─metrics           # evaluation metrics computing
├─models            # Model, network modules, and loss design
├─utils             # utilities
├─main.py           # MtsCID main entrance.
├─checkpoints       # model checkpoint
└─results           # expeirmental results.        

```

## Environment

**Key Packages:**

PyTorch v1.11.0 + (cu11.3)

python v3.8.6

scikit-learn


## Data

In this repository, the GECCO dataset under 100 time steps setting is proposed for a quick hands-up.

## Usage

The simplest way of running MtsCID is to run `python main.py`.

## Citation
If you find this repo useful, please cite our paper. 
```bibtex
@article{xie2025multivariate,
  title={Multivariate Time Series Anomaly Detection by Capturing Coarse-Grained Intra-and Inter-Variate Dependencies},
  author={Xie, Yongzheng and Zhang, Hongyu and Babar, Muhammad Ali},
  journal={arXiv preprint arXiv:2501.16364},
  year={2025}
}
```
