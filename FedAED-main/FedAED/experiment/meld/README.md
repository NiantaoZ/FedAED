
### How to run MELD Dataset in FedMultimodal (Audio and Text)
Here we provide an example to quickly start with the experiments, and reproduce the Meld results from the paper. We set the fixed seed for data partitioning, training client sampling.

#### 0. Download data: The data will be under data/meld by default.

You can modify the data path in system.cfg to the desired path.

```
cd data
bash download_meld.sh
cd ..
```

Data will be under data/meld, and you will need about 20GB space

#### 1. Partition the data

This data has a natural partition (Speaker ID).

```
python3 features/data_partitioning/meld/data_partition.py
```

The return data is a list, each item containing [key, file_name, label, speaker_id, utterance text].

#### 2. Feature extraction

For MELD dataset, the feature extraction includes text/audio feature extraction.

```
# extract mfcc (audio) feature
taskset -c 1-30 python3 features/feature_processing/meld/extract_audio_feature.py --feature_type mfcc

# extract mobilebert feature
taskset -c 1-30 python3 features/feature_processing/meld/extract_text_feature.py --feature_type mobilebert
```

#### 3. (Optional) Simulate missing modality conditions

default missing modality simulation returns missing modality at 10%, 20%, 30%, 40%, 50%

```
cd features/simulation_features/meld
# output/mm/meld/{client_id}_{mm_rate}.json

# missing modalities
bash run_mm.sh
cd ../../../
```
The return data is a list, each item containing:
[missing_modalityA, missing_modalityB, new_label, missing_label]

missing_modalityA and missing_modalityB indicates the flag of missing modality, new_label indicates erroneous label, and missing label indicates if the label is missing for a data.

#### 4. Run base experiments (FedAvg, FedOpt, FedProx, ...)
```
cd experiment/meld
bash run_base.sh
```

#### 5. Run ablation experiments, e.g Missing Modality (the same for missing labels and label noises)
```
cd experiment/meld
bash run_mm.sh
```

