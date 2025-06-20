# Introducing the AMD-EfficientVLM (Vision-Language Model):

⚡️ This repository provides training recipes for the AMD efficient vision-language models, which are designed to improve the inference efficiency of VLM. 

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

You can run pre-training using scripts/pretrain.sh. The mm projector is pre-trianing during this stage. Please set the data_path and image_folder according to your data path. 
```
bash scripts/pretrain.sh
```

### Training

You can use follwing command to start training the model.

```
bash scripts/train.sh
```

### Supervised Fine-tuning

You can use following command to start supervised fine-tuning the model.

```
bash scripts/sft.sh
```

### Inference
We provide a simple script for inference with a single image input.

```
python3 llava/test_generate.py --model_path ./checkpoints/ --image_path ./images/dino.png --question "Please describe this image."
```


## Call to Action
You are welcome to download and try this model. To get more information about the training, inferencing and insights of this model, please visit the AMD Hugging Face Model Card to get access to the codes, and to download the model file. Additionally, AMD opened a dedicated cloud infrastructure that includes latest GPU instances to AI developers. Visit [AMD Developer Cloud](https://www.amd.com/en/forms/registration/developer-cloud-application.html) for specific accessing request and usage. Furthermore, you can deploy advanced AI models on AMD Ryzen AI PCs and can learn more [here](https://www.amd.com/en/developer/resources/ryzen-ai-software.html).

For any questions, you may reach out to the AMD team at amd_ai_mkt@amd.com.
