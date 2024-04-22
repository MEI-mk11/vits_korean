import whisper
import os



model = whisper.load_model("base")

def transcribe_one(file_path):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    # _, probs = model.detect_language(mel)
    # print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # print the recognized text
    return result.text

folder_path = 'data/korean2/'

for root, dirs, files in os.walk(folder_path):
    for file in files:

        file_path = os.path.join(root, file)


        rec_result = transcribe_one(file_path)
        content = file_path +'||0||' + rec_result + '\n'
        with open('data/esd.txt', "a") as file:
            file.write(content)