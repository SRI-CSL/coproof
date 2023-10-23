"""Applies simplet5 training.

Be sure to run 220325_data_setup.py first to featurize the proof data
into the correct form.

"""
import pandas as pd
from coprover.training.simplet5 import SimpleT5
from sklearn.model_selection import train_test_split
from pathlib import Path


USE_MLM = False
TRAIN = True

# Command prefixes are always expected
CMD1_PREFIX = "command1: "

full_df = pd.read_csv("tags_input_pred_pairs.tsv",
                      sep="\t",
                      header=None,
                      names=['source_text',
                             'target_text'])

full_df['source_text'] = CMD1_PREFIX + full_df['source_text']

train_df, test_df = train_test_split(full_df, test_size=0.1,
                                     random_state=1337,
                                     shuffle=True)

# Get max on sentence lengths
max_src_tok_len = max([len(x.split()) for x in full_df['source_text']]) + 10

print("Max source toklength={}".format(max_src_tok_len))

# Prediction task, use minimal
model = SimpleT5(source_max_token_len=max_src_tok_len,
                 target_max_token_len=10)
model.from_pretrained("t5", "t5-base")

# Load in existing pre-trained model
if USE_MLM:
    print("Loading  model from MLM N1")
    MLM_MODEL = Path("models", "mlm_N1", "curr_best")
    model.load_model(MLM_MODEL)


# CACHED_FPATH = Path("models", "cmdprec_mlmn1", "curr_best")
CACHED_FPATH = Path("models", "cmdprec", "curr_best")
if TRAIN:
    model.train(train_df=train_df,
                eval_df=test_df,
                max_epochs=10,
                batch_size=4,
                dataloader_num_workers=4,
                outputdir="outputs/cmd_pred1_updated",
                save_only_last_epoch=True,
                num_gpus=2)
else:
    print("Using existing cmdpred model at {}".format(CACHED_FPATH))
    model.load_model(model_dir=CACHED_FPATH)
    


print(model.predict(CMD1_PREFIX + "<ANT> <CONS> s-formula forall ['variable'] ['variable'] apply constant type-actual apply constant type-actual type-actual apply constant ['variable'] ['variable'] apply constant apply constant type-actual type-actual ['variable'] apply constant type-actual type-actual ['variable'] <HID>"))


from tqdm import tqdm

def score_df(df):
    num_correct = 0
    total = 0
    guesses = []
    golds = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        guess = model.predict(row['source_text'])[0]
        gold = row['target_text']
        guesses.append(guess)
        golds.append(gold)
        total += 1
        if guess == gold:
            num_correct += 1
    return num_correct / total

print("Train acc={:.3f}".format(score_df(train_df)))
print("Test acc={:.3f}".format(score_df(test_df)))

# TODO Confusion
