# PruneQuant-TrafClassif

PruneQuant-TrafClassif performs efficient traffic classification using the ISCXVPN2016, USTC-TFC2016, CICIOT2022, and ITC-NET-AUDIO-5 dataset. Follow these steps to get started. Change dataset argument from `iscx2016vpn`  to `ustctfc2016` or `ciciot2022` or `itcnetaudio5` accordingly.


&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

## 1 - Dataset Preparation

1. **Download Dataset**:
   - Get ISCX2016VPN from [ISCXVPN2016](https://www.unb.ca/cic/datasets/vpn.html).
   - Get USTC-TFC2016 from [USTC-TFC2016](https://github.com/davidyslu/USTC-TFC2016).
   - Get CICIOT2022 from [CICIOT2022](https://www.unb.ca/cic/datasets/iotdataset-2022.html).
   - Get ITC-NET-AUDIO-5 from [ITC-NET-AUDIO-5](https://bmcresnotes.biomedcentral.com/articles/10.1186/s13104-024-06718-7).

2. **Extract PCAP Files**:
   Extract `.pcap` and `.pcappng` files into `repo/data/datasets/iscx2016vpn-extracted-pcaps/`

3. **Preprocess Data**:
   Run `python utils/preprocess_iscx2016vpn.py` or `python utils/preprocess_ustc-tfc2016.py` to preprocess packet files to DL friendly format.

&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;


## 2 - Training

Execute the training script:
```bash
python utils/training_utils.py --dataset iscx2016vpn --NiN_model false --epochs 250 --batch_size 128
```

&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;



## 3 - Search for best pruning and quantization strategy
To prune 20% of the filters and quantize to 8 bit, execute:
```bash
python auto_prune_quant.py --dataset iscx2016vpn --model CNN1D_TrafficClassification --prune_ratio 0.2 --qat_epochs 5 --data_bsize 1024 --seed 2024 --action_std 0.5 --max_episodes 500
```
<!-- python auto_prune_quant.py --dataset iscx2016vpn --model CNN1D_TrafficClassification --prune_ratio 0.2 --qat_epochs 5 --data_bsize 1024 --seed 2024 --action_std 0.5 --max_episodes 1000 | tee logs/iscx2016vpn/prune0.20_qatepch5_bsize1024_actionstd0.5_maxepsd1000.txt -->

&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;


## 4 - Finetuning and Inferencing

Improve model accuracy by finetuning with optimal settings. This generates an ONNX model.
```bash
python prune_quant/finetune.py --dataset iscx2016vpn --epochs 25 --lr 0.001 --batch_size 32
```
Convert the ONNX model to a TensorRT (.trt) engine:
<!-- Use new terminal, cd to repo -->
<!-- export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zafri/anaconda3/envs/cuda12:/home/zafri/anaconda3/envs/cuda12/lib -->
<!-- /home/zafri/anaconda3/envs/cuda12/TensorRT-8.6.1.6/bin/trtexec --onnx=networks/quantized_models/iscx2016vpn/model.onnx --saveEngine=networks/quantized_models/iscx2016vpn/model_engine.trt --int8 -->
```bash
trtexec --onnx=networks/quantized_models/iscx2016vpn/model.onnx --saveEngine=networks/quantized_models/iscx2016vpn/model_engine.trt --int8 
```
Evaluate the performance and accuracy of the TensorRT engine (use a batch size comparable to that used during fine-tuning):
```bash
python prune_quant/finetune.py --batch_size 1 --eval_trt true --dataset iscx2016vpn
```
<!-- To capture memory vs cpu breakdown: -->
<!-- python prune_quant/finetune.py --batch_size 1 --eval_trt true --dataset iscx2016vpn --trt_mem_cpu_breakdown true -->
