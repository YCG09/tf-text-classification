CNN and RNN with Attention for Text Classification in Tensorflow

## Requirements

* Python 3
* Tensorflow 1.0
* Scikit-learn
* Numpy
* Pandas

## Training models
CNN: 
```bash
python train_cnn.py
```
RNN:
```bash
python train_rnn.py
```

## Evaluating models
```bash
python eval.py --checkpoint_dir=./runs/textrnn/trained_result_1512462690/checkpoints --model_type=RNN
```
If evaluation data has labels, `has_label` should be set to `True`.
