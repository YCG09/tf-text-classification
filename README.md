## Environment setup
* Requirements: python3, tensorflow>=1.0, sklearn, numpy, pandas and jieba
```bash
pip install -r requirements.txt
```

## Data preprocessing
* Text cleaning and word segmentation, data format: label\tsentence
```bash
python seg_words.py
```

## Training models
* CNN
```bash
python train_cnn.py
```
* RNN with attention
```bash
python train_rnn.py
```

## Evaluating models
```bash
python eval.py --checkpoint_dir=./runs/textrnn/trained_result_1512462690/checkpoints --model_type=RNN
```
If evaluation data has labels, `has_label` should be set to `True`.
