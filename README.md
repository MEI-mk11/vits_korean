Here is a finetune version of vits korean. The basic pretrain model is provided by https://github.com/0913ktg/vits_korean_multispeaker

## Process of Finetuning

Run modifiy_sr.py to detect or modify the sampling rate to 22050, remember to change the input folder and output folder manually

Use autolabel.py to auto-recognize text and convert it to the required form

Run preprocess.py to clean the labelled text

Use make_mels.py to generate mel features

Run train_ms to train, you can edit the config file to fit your needs
