# FedAED
## Federated Adaptive Equilibrium Distillation for Effective Cross-Modal Knowledge Transfer in Multimodal Federated Learning
### About the research
In this study, we propose a general Multimodal Federated Learning (MFL) framework called Federated Adaptive Equilibrium Distillation (FedAED) to address the challenges of modality heterogeneity, including missing modalities, incomplete labels, and annotation errors. FedAED enhances cross-modal collaboration by dynamically adapting knowledge transfer paths and establishing a globally consistent semantic space, enabling personalized local models to effectively learn complementary multimodal information while preserving robustness under dynamic modality availability. This framework improves the effectiveness, stability, and fairness of MFL under diverse heterogeneous conditions.

<img width="4042" height="1936" alt="framework" src="https://github.com/user-attachments/assets/6f6c8eaa-2080-44db-87b7-787f014c659f" />


To overcome the limitations of static knowledge transfer designs that lack flexibility to adapt to changing modality availability, FedAED introduces three complementary components: Adaptive Modality Alignment (AMA), Cross-modal Contrastive Regularization (CCR), and Asynchronous Bi-directional Collaboration (ABC).
- **Adaptive Modality Alignment (AMA)** adopts a gated attention mechanism adapted to MFL environments. It dynamically adjusts knowledge propagation paths based on real-time modality availability, refining each modality’s representation by aligning it with complementary cues from other available modalities.
- **Cross-modal Contrastive Regularization (CCR)** captures complementary associations across modalities through dual-level contrastive learning. It promotes intra-modal consistency and inter-modal alignment, bridging semantic gaps across heterogeneous clients.
- **Asynchronous Bi-directional Collaboration (ABC)** maintains separate teacher and student knowledge buffers with a time-decay aggregation strategy. This design mitigates noisy updates from asynchronous client participation while preserving the integrity of global knowledge.

This triple mechanism ensures that local models continuously adapt to missing or corrupted modalities while synchronizing with global representations, allowing both cross-modal generalization and local robustness to be effectively maintained throughout the training process. Theoretical analysis confirms that FedAED guarantees complete knowledge transfer even under missing modalities. Comprehensive experiments conducted on five benchmark datasets covering six modalities (text, image, audio, physiological signals) under three types of data corruptions (missing modalities, missing labels, erroneous labels) demonstrate that FedAED consistently outperforms state-of-the-art MFL methods in accuracy, robustness, and fairness.

For implementation details, model design, and experimental configurations, please refer to our paper entitled: “Federated Adaptive Equilibrium Distillation for Multimodal Federated Learning under Modality Heterogeneity.”

## Applications supported

* #### Cross-Device Applications
    * Human Activity Recognition [[UCI-HAR](https://github.com/usc-sail/fed-multimodal/tree/main/fed_multimodal/experiment/uci-har)] [[KU-HAR](https://github.com/usc-sail/fed-multimodal/tree/main/fed_multimodal/experiment/ku-har)]
    * Social Media [[Crisis-MMD](https://github.com/usc-sail/fed-multimodal/tree/main/fed_multimodal/experiment/crisis-mmd)]
    * Emotion Recognition [[MELD](https://github.com/usc-sail/fed-multimodal/tree/main/fed_multimodal/experiment/meld)]
* #### Cross-silo Applications (e.g., Medical Settings)
    * ECG classification [[PTB-XL](https://github.com/usc-sail/fed-multimodal/tree/main/fed_multimodal/experiment/ptb-xl)]
 
## Data processing recipe

Feature processing includes 3 steps:

* Data partitioning
* Simulation features (missing modalities, missing labels, erroneous labels)
* Feature processing

## Quick Start – UCI-HAR Example (Acc. + Gyro)

Here we provide an example to quickly start with the experiments, and reproduce the UCI-HAR results with FedAED. We set fixed seeds for data partitioning and client sampling.

### 0. Download data

The data will be under `data/uci-har` by default. You can modify the data path in `system.cfg` if needed.
`cd FedAED/data`
`bash download_uci_har.sh`
`cd ..`

### 1. Partition the data

`alpha` specifies the non-IIDness of the partition; lower values produce higher data heterogeneity. Each subject’s data is partitioned into sub‑clients.

`python3 features/data_partitioning/uci-har/data_partition.py --alpha 0.1 --num_clients 20`
`python3 features/data_partitioning/uci-har/data_partition.py --alpha 5.0 --num_clients 20`

### 2. Feature extraction
The returned data is a list, each item containing `[key, file_name, label]`.
For UCI-HAR, feature extraction mainly handles normalization.

`python3 features/feature_processing/uci-har/extract_feature.py --alpha 0.1`
`python3 features/feature_processing/uci-har/extract_feature.py --alpha 5.0`

### 3. (Optional) Simulate missing modality conditions

Default missing modality simulation returns missing rates at 10%, 20%, 30%, 40%, 50%.
'cd features/simulation_features/uci-har'

The returned data is a list, each item containing:  
`[missing_modalityA, missing_modalityB, new_label, missing_label]`.  
`missing_modalityA` and `missing_modalityB` indicate missing modality flags, `new_label` indicates erroneous label, and `missing_label` indicates if the label is missing for a data point.

### 4. Run baseline experiments (FedAvg, FedOpt, FedProx, ...)
`cd experiment/uci-har`
`bash run_base.sh`

### 5. Run FedAED experiments
To run the proposed FedAED framework (which includes AMA, CCR, and ABC):
`cd experiment/uci-har`
`bash run_fedaed.sh`


You can also run FedAED with specific corruption settings (missing modalities, missing labels, erroneous labels, or combined):
`python main.py --method FedAED --dataset uci-har --miss_modality_rate 0.3`
`python main.py --method FedAED --dataset uci-har --miss_label_rate 0.2`
`python main.py --method FedAED --dataset uci-har --err_label_rate 0.2`
`python main.py --method FedAED --dataset uci-har --miss_modality_rate 0.3 --miss_label_rate 0.2 --err_label_rate 0.2`

## Citation

If you would like to obtain more detailed information, please refer to the original benchmark paper:

```bibtex
@article{feng2023fedmultimodal,
  title={FedMultimodal: A Benchmark For Multimodal Federated Learning},
  author={Feng, Tiantian and Bose, Digbalay and Zhang, Tuo and Hebbar, Rajat and Ramakrishna, Anil and Gupta, Rahul and Zhang, Mi and Avestimehr, Salman and Narayanan, Shrikanth},
  journal={arXiv preprint arXiv:2306.09486},
  year={2023}
}
