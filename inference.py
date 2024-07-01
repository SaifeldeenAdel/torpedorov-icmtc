import os
import pandas as pd
import tensorflow as tf
import keras

from modelArch import Transformer


#######################################
# INFERENCE TAKES ABOUT 4 HOURS TO RUN
#######################################


## Vectorizer 
class VectorizeChar:
    def __init__(self, max_len=50):
        self.vocab = (
                        ["-", "#", "<", ">"]
                        + ['آ', 'إ', 'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ'
                            , 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ',
                            'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'أ', 'ؤ'
                            , 'ئ', 'ة', 'ى', 'ء', ' ', 'f','o','l', "،", "ٱ","ڨ", "چ"]
                        + [" ", ".", ",", "?"]
        )
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        text = text.lower()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab


    
# Define the audio processing function
def path_to_audio(path):
    # spectrogram using stft
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1)
    audio = tf.squeeze(audio, axis=-1)
    stfts = tf.signal.stft(audio, frame_length=200, frame_step=80, fft_length=256)
    x = tf.math.pow(tf.abs(stfts), 0.5)
    mx=tf.math.reduce_max(x)
    # normalisation
    means = tf.math.reduce_mean(x, 1, keepdims=True)
    stddevs = tf.math.reduce_std(x, 1, keepdims=True)
    epsilon = 1e-7  # Very small number
    stddevs = tf.where(stddevs == 0, epsilon, stddevs)  # Replace zero stddevs with epsilon
    
    x = (x - means) / stddevs

    audio_len = tf.shape(x)[0]
    # padding to 10 seconds
    pad_len = int(2754*1.5)
    paddings = tf.constant([[0, pad_len], [0, 0]])
    x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
    return x


# Function to get file names without extension
def get_file_id(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]


# Load the model
model = keras.models.load_model(
    "./my_model_v3.keras",
    compile=False,
    custom_objects={"Transformer": Transformer}
)

# Directory containing the .wav files
audio_directory = "./test"

# Prepare to make predictions
target_start_token_idx = 2
target_end_token_idx = 3
idx_to_char = VectorizeChar().get_vocabulary()
results = []

cnt = 0
# Iterate over the .wav files and make predictions
for audio_file in os.listdir(audio_directory):
    if audio_file.endswith(".wav"):
        file_path = os.path.join(audio_directory, audio_file)
        source = path_to_audio(file_path)
        source = tf.expand_dims(source, 0)  # Add batch dimension
        preds = model.generate(source, target_start_token_idx)
        preds = preds.numpy()
        prediction = ""
        for idx in preds[0, :]:
            prediction += idx_to_char[idx]
            if idx == target_end_token_idx:
                break
        audio_id = get_file_id(file_path)
        print(f"Predicting sample {cnt}..")
        results.append({"audio": audio_id, "transcript": prediction[1:-1]})
        cnt += 1

# Save the results to a new CSV file
output_df = pd.DataFrame(results)
output_df.to_csv("submission.csv", index=False)

print(f"Predictions saved to submission.csv")