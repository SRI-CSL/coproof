# --> Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

path = "https://raw.githubusercontent.com/Shivanandroy/T5-Finetuning-PyTorch/main/data/news_summary.csv"
df = pd.read_csv(path)


# --> preprocessing dataset: training_df, test_df with "source_text" & "target_text" columns

# simpleT5 expects dataframe to have 2 columns: "source_text" and "target_text"
df = df.rename(columns={"headlines":"target_text", "text":"source_text"})
df = df[['source_text', 'target_text']]

# T5 model expects a task related prefix: since it is a summarization task, we will add a prefix "summarize: "
df['source_text'] = "summarize: " + df['source_text']

train_df, test_df = train_test_split(df, test_size=0.2)


# --> Finetuning T5 model with simpleT5

from simplet5 import SimpleT5

model = SimpleT5()
model.from_pretrained(model_type="t5", model_name="t5-base")
model.train(train_df=train_df,
            eval_df=test_df, 
            source_max_token_len=128, 
            target_max_token_len=50, 
            batch_size=8, max_epochs=3, use_gpu=True)


# --> Load and inference

# let's load the trained model for inferencing:
model.load_model("t5","outputs/simplet5-epoch-2-train-loss-0.9074-val-loss-1.0378", use_gpu=True)

text_to_summarize="""summarize: Rahul Gandhi has replied to Goa CM Manohar Parrikar's letter, 
which accused the Congress President of using his "visit to an ailing man for political gains". 
"He's under immense pressure from the PM after our meeting and needs to demonstrate his loyalty by attacking me," 
Gandhi wrote in his letter. Parrikar had clarified he didn't discuss Rafale deal with Rahul.
"""
model.predict(text_to_summarize)

# --> model quantization & ONNX support

# for faster inference on cpu, quantization, onnx support:
model.convert_and_load_onnx_model(model_dir="outputs/SimpleT5-epoch-2-train-loss-0.9526")
model.onnx_predict(text_to_summarize)
