# PruneQuant-TrafClassif

PruneQuant-TrafClassif simplifies traffic classification using the ISCXVPN2016 dataset. Follow these steps to get started.

## Dataset Preparation

1. **Download Dataset**: Get ISCX2016VPN from [UNB datasets](https://www.unb.ca/cic/datasets/vpn.html). Cite the dataset using the provided reference.

2. **Extract PCAP Files**:
   Extract `.pcap` or `.pcappng` files into `repo/data/datasets/iscx2016vpn-extracted-pcaps/`

3. **Preprocess Data**:
   Run `python utils/preprocess_iscx2016vpn.py` to preprocess packet files to DL friendly format.

## Training

Execute the training script. Adjust --epochs and --batch_size as needed.
```bash
python utils/iscx2016vpn_training_utils.py --NiN_model false --epochs 120 --batch_size 1024
