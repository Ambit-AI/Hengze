import os
os.environ['CUDA_LAUNCH_BLOCKING']="1"
from datasets import load_dataset
dataset = load_dataset("tweet_eval", "sentiment")
train_data, valid_data, test_data = dataset['train'], dataset['validation'], dataset['test']

from run_Roberta_model import model_train_validate_test
import pandas as pd
from utils import Metric
import os
target_dir = "./output2"

model_train_validate_test(train_data, valid_data, test_data, target_dir,
         max_seq_len=128,
         epochs=15,
         batch_size=32,
         lr=2e-05,
         patience=10,
         max_grad_norm=10.0,
         if_save_model=True,
         checkpoint=None)
# test_result = pd.read_csv(os.path.join(target_dir, 'test_prediction.csv'))
# Metric(test_data['label'], test_result.prediction)