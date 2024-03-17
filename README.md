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
python utils/iscx2016vpn_training_utils.py --NiN_model false --epochs 250 --batch_size 128
```




## Finetuning and Inferencing

Improve model accuracy by finetuning with optimal settings. Adjust parameters such as batch size (--batch_size), epochs (--epochs), and learning rate (--lr). This generates an ONNX model.
```bash
python prune_quant/finetune.py --epochs 25 --lr 0.001 --batch_size 32
```
Convert the ONNX model to a TensorRT (.trt) engine:
```bash
trtexec --int8 --onnx=networks/quantized_models/iscx2016vpn/model.onnx --saveEngine=networks/quantized_models/iscx2016vpn/model_engine.trt
```
Evaluate the performance and accuracy of the TensorRT engine (use same batch_size as finetuning):
```bash
python prune_quant/finetune.py --batch_size 32 --eval_trt true
```
