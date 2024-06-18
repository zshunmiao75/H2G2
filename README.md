# Enhancing Entity Relationship Extraction in Dialogue Texts using Hypergraph and Heterogeneous Graph



## Environments

- python		(3.8.3)
- cuda			(11.0)

## Requirements

- dgl-cu110			   (0.5.3)
- torch					   (1.7.0)
- numpy					(1.19.2)
- sklearn
- regex
- packaging
- tqdm


## Usage

- run_classifier.py : Code to train and evaluate the model
- data.py : Code to define Datasets / Dataloader for HG2G 
- evaluate.py : Code to evaluate the model on DialogRE
- models/BERT : The directory containing the HG2G for BERT version
- models/RoBERTa : The directory containing the HG2G  for RoBERTa version


## Preparation

### Dataset

#### DialogRE

- Download data from [here](https://github.com/nlpdata/dialogre) 
- Put `train.json`, `dev.json`, `test.json` from ```data_v2/en/data/``` into the directory `datasets/DialogRE/`



### Pre-trained Language Models

#### BERT Base

- Download and unzip BERT-Base Uncased from [here](https://github.com/google-research/bert), and copy the files into the directory `pre-trained_model/BERT/`
- Set up the environment variable for BERT by ```export BERT_BASE_DIR=/PATH/TO/BERT/DIR```. 
- In `pre-trained_model`, execute ```python convert_tf_checkpoint_to_pytorch_BERT.py --tf_checkpoint_path=$BERT_BASE_DIR/bert_model.ckpt --bert_config_file=$BERT_BASE_DIR/bert_config.json --pytorch_dump_path=$BERT_BASE_DIR/pytorch_model.bin```.

#### RoBERTa Large

- Download and unzip RoBERTa-large from [here](https://github.com/pytorch/fairseq/tree/master/examples/roberta), and copy the files into the directory `pre-trained_model/RoBERTa/`
- Download `merges.txt` and `vocab.json` from [here](https://huggingface.co/roberta-large/tree/main) and put them into the directory `pre-trained_model/RoBERTa/`
- Set up the environment variable for RoBERTa by ```export RoBERTa_LARGE_DIR=/PATH/TO/RoBERTa/DIR```. 
- In `pre-trained_model`, execute ```python convert_roberta_original_pytorch_checkpoint_to_pytorch.py --roberta_checkpoint_path=$RoBERTa_LARGE_DIR --pytorch_dump_folder_path=$RoBERTa_LARGE_DIR```.

## Training & Evaluation

### BERT + DialogRE

- Execute the following commands in ```HG2G ```:

```
python run_classifier.py --do_train --do_eval --encoder_type BERT  --data_dir datasets/DialogRE --data_name DialogRE   --vocab_file $BERT_BASE_DIR/vocab.txt   --config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 12   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir TUCOREGCN_BERT_DialogRE  --gradient_accumulation_steps 2

rm HG2G_BERT_DialogRE/model_best.pt

python evaluate.py --dev datasets/DialogRE/dev.json --test datasets/DialogRE/test.json --f1dev HG2G_BERT_DialogRE/logits_dev.txt --f1test HG2G_BERT_DialogRE/logits_test.txt --f1cdev HG2G_BERT_DialogRE/logits_devc.txt --f1ctest TUCOREGCN_BERT_DialogRE/logits_testc.txt --result_path HG2G_BERT_DialogRE/result.txt
```




### RoBERTa + DialogRE

- Execute the following commands in ```HG2G```:

```
python run_classifier.py --do_train --do_eval --encoder_type RoBERTa  --data_dir datasets/DialogRE --data_name DialogRE   --vocab_file $RoBERTa_LARGE_DIR/vocab.json --merges_file $RoBERTa_LARGE_DIR/merges.txt  --config_file $RoBERTa_LARGE_DIR/config.json   --init_checkpoint $RoBERTa_LARGE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 12   --learning_rate 7.5e-6   --num_train_epochs 30.0   --output_dir TUCOREGCN_RoBERTa_DialogRE  --gradient_accumulation_steps 2

rm HG2G_BERT_DialogRE/model_best.pt

python evaluate.py --dev datasets/DialogRE/dev.json --test datasets/DialogRE/test.json --f1dev HG2G_BERT_DialogRE/logits_dev.txt --f1test HG2G_BERT_DialogRE/logits_test.txt --f1cdev HG2G_BERT_DialogRE/logits_devc.txt --f1ctest TUCOREGCN_RoBERTa_DialogRE/logits_testc.txt --result_path HG2G_BERT_DialogRE/result.txt
```




```
