import os

import librosa
import soundfile as sf

def get_audio_sample_rate(file_path):
    try:
        # 使用librosa加载音频文件
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        return sample_rate, audio_data
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

file_path = 'Korean-Read-Speech-Corpus/dataset/AirbnbStudio'
output_path = 'data/korean2'
def modify_sampling_rate(file_path, output_path):

    for file in os.listdir(file_path):
        # if file.endswith('.flac'):
        file_total = os.path.join(file_path, file)

        if file_total:

            sample_rate, audio_data = get_audio_sample_rate(file_total)
            if sample_rate is not None:
                print(f"The audio sample rate is: {sample_rate} Hz")
            else:
                print("Failed to retrieve the audio sample rate.")

            if sample_rate != 22050:
                print('need resample')
                sample_rate_resampled = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=22050)
                file = file.replace('.flac', '.wav')
                output_path_final = os.path.join(output_path,file)
                sf.write(output_path_final, sample_rate_resampled, 22050)

        else:

            continue

    return 'finished'

result = modify_sampling_rate(file_path, output_path)
print(result)

# sample_rate, audio_data = get_audio_sample_rate("/nfs/meizhengkun/vits_korean_multispeaker/Korean-Read-Speech-Corpus/dataset/AirbnbStudio/sub100100a00000.wav")
# print(sample_rate)