CNN and RNN with Attention for Text Classification in Tensorflow

## Requirements

* Python 3
* Tensorflow >= 1.0
* Scikit-learn
* Numpy
* Pandas

## Train
CNN model: 
```bash
python train_cnn.py
```
RNN model:
```bash
python train_rnn.py
```

## Evaluate
```bash
python eval.py --checkpoint_dir=./runs/textrnn/trained_result_1512462690/checkpoints --model_type=RNN
```
