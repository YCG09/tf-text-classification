## Environment setup
* Requirements: python3, tensorflow1.10, sklearn, numpy, pandas and jieba
```bash
pip install -r requirements.txt
```

## Training
* CNN
```bash
python train_cnn.py
```
* RNN with attention
```bash
python train_rnn.py
```
#### the data format is as follows: label '\t' text

## Testing
* evaluation mode:
```bash
python test.py --checkpoint_dir=./runs/textcnn/trained_result_1548144557/checkpoints --model_type=CNN
```
* prediction mode:
```bash
python test.py --checkpoint_dir=./runs/textcnn/trained_result_1548144557/checkpoints --model_type=CNN --test_mode=prediction
```

## Inference
```bash
python inference.py --checkpoint_dir=./runs/textrnn/trained_result_1548145204/checkpoints --model_type=RNN
```

