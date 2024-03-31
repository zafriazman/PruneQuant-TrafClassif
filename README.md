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

## Search for best pruning and quantization strategy
To remove 60% of the filters, execute
```bash
python auto_prune_quant.py --dataset iscx2016vpn --model CNN1D_TrafficClassification --prune_ratio 0.6 --qat_epochs 1 --data_bsize 1024 --max_episodes 200
python auto_prune_quant.py --dataset iscx2016vpn --model CNN1D_TrafficClassification --prune_ratio 0.1 --qat_epochs 5 --data_bsize 1024 --seed 2024 --action_std 0.3 | tee logs/prune0.1_qatepch5_bsize1024_actionstd0.3.txt
```




## Finetuning and Inferencing

Improve model accuracy by finetuning with optimal settings. Adjust parameters such as batch size (--batch_size), epochs (--epochs), and learning rate (--lr). This generates an ONNX model.
```bash
python prune_quant/finetune.py --epochs 25 --lr 0.001 --batch_size 32
```
Convert the ONNX model to a TensorRT (.trt) engine:
```bash
trtexec --onnx=networks/quantized_models/iscx2016vpn/model.onnx --saveEngine=networks/quantized_models/iscx2016vpn/model_engine.trt --int8 
```
Evaluate the performance and accuracy of the TensorRT engine (use a batch size comparable to that used during fine-tuning):
```bash
python prune_quant/finetune.py --batch_size 1 --eval_trt true
```
