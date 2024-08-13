# DAISY: Data Adaptive Self Supervised Early Exit on Speech Representation Models
This is official implementation of Interspeech 2024 paper: [DAISY: Data Adaptive Self Supervised Early Exit on Speech Representation Models](https://arxiv.org/abs/2406.05464)

# Preparation 
1. Please install the required package of [s3prl](https://github.com/s3prl/s3prl)
2. Please copy the files that need to be changed into the s3prl folder.
    ```
    cp -r s3prl_daisy/s3prl/* s3prl/
    ```
3. Please download the checkpoint of DAISY from [link](https://drive.google.com/file/d/1-r3KUoOt-zsd6XDasBSWzZXvBom8a_f9/view?usp=sharing). This weight of this model is 100\% identical to HuBERT base except that it has pretrained early exit branches at each layer.
4. Run downstream task with dynamically early exit on modified s3prl. The following is an example of speaker identification. Please refer to [s3prl-note](https://github.com/s3prl/s3prl/blob/master/s3prl/downstream/docs/superb.md) to know how to run other downstream tasks.
    ```
    python3 run_downstream.py -m train -u ee_hubert_local_cluster -d voxceleb1 -n example -k [DAISY_CHECKPOINT] --featurizer_type dynamic --upstream_feature_normalize --upstream_model_config upstream/ee_hubert/downsteam_config/sid.yaml --upstream_log example.txt
    ```
    Note: You should change *--upstream_model_config*, *-d*, *--upstream_log* for different downstream tasks.
    DAISY_CHECKPOINT: Could be downloaded from [link](https://drive.google.com/file/d/1-r3KUoOt-zsd6XDasBSWzZXvBom8a_f9/view?usp=sharing)
