# Song2Mood
Song Mood Classification(Happy, Aggressive, Sad, Calm)

## Environment
* python3
```
pip install -r requirements.txt
```

## Data
* Training: 16000
* Testiong: 1000

### Data Preprocessing
* constant-Q transform

## Model
* 實做 [Learning to Recognize Transient Sound Events Using Attentional Supervision](https://www.ijcai.org/proceedings/2018/0463.pdf)

### Train
```
$ python -m synet.train
```
### Evaluation
```
$ python -m synet.test_data_result

```
matrices          | Happy  | Aggressive | Sad | Calm
--------------|:-----:|-----:| ----:|-----------------
f1-score   | 0.843 |  0.681 | 0.866 | 0.824
recall   | 0.868 | 0.665 | 0.881 | 0.845
precision   | 0.819 | 0.698 | 0.85 | 0.805