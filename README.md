# Introducing the First AMD Efficient VLM (Vision-Language Model): AMD-EVLM

Core Contributors: Zhenhua Liu, Xuanwu Yin, Dong Li, Barsoum Emad

License Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

⚡️ This repository provides training recipes for the AMD efficient vision-language models, which are designed for high performance and efficiency. The training process includes three key steps:

Pre-training, Finetuing and Supervised Finetuning: We use a LORA-like method to compress and accelerate the popular vision-language model LLaVA-OneVision, achieving a competitive performance with only about only 1/6 FLOPs.

This implementation is released to promote further research and innovation in the field of efficient vision-language models, optimized for AMD Instinct GPUs.


We trained the models on a cluster of AMD Instinct<sup>TM</sup> MI250 GPUs and achieved a comparable performane with much less computation costs.

|Models|GFLOPs|MMBench|MME|MMMU|ScienceQA|TextVQA|MMVet|
|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Ivy-VL|2.2x10^4|82.6|1951|46.7 |97.3|76.48|44.12|
|Ours|0.37x10^4|82.4| 1943 |46.2 |97.1 |76.27 |43.54|


## Getting Started 

### Prepare environment 
You can use the following command to install necessary packages.

```
pip3 install -r requirements.txt
pip3 install flash-attn --no-build-isolation
```

### Prepare training data
You can follow to [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main/scripts/train#about-the-llava-onevision-data) to prepare the pre-training and the finetuning data.

### Pre-training

You can run pre-training using scripts/pretrain.sh. Please set the data_path and image_folder according to your data path. 
```
bash scripts/pretrain.sh
```

### Finetuning

You can use follwing command to start fine-tuning the model. The checkpoint will be saved at checkpoints/EVLM-3b-finetune.

```
bash scripts/finetune.sh
```

### Supervised Fine-tuning

You can use following command to start supervised fine-tuning the model.

```
bash scripts/sft.sh
```

### Inference
We provide a simple script for inference with a single image input.

```
python3 llava/test_generate.py --model_path ./checkpoints/vlora-7b-sft --image_path ./images/dino.png --question "Please describe this image."
```


## Call to Action
You are welcome to download and try this model. To get more information about the training, inferencing and insights of this model, please visit the AMD Hugging Face Model Card to get access to the codes, and to download the model file. Additionally, AMD opened a dedicated cloud infrastructure that includes latest GPU instances to AI developers. Visit [AMD Developer Cloud](https://www.amd.com/en/forms/registration/developer-cloud-application.html) for specific accessing request and usage. Furthermore, you can deploy advanced AI models on AMD Ryzen AI PCs and can learn more [here](https://www.amd.com/en/developer/resources/ryzen-ai-software.html).

For any questions, you may reach out to the AMD team at amd_ai_mkt@amd.com.