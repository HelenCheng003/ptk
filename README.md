# ptk
A CNN Classifier using Spectrograms

## Warning: Bugs are not yet fixed
The Classifier cannot do a proper prediction as desired.

## Algorithm to tackle the task
### Step 1: Split `ptk_2.wav` into `.wav` files only include a Pa/Ta/Ka sound by `pydub.slience`

Why this can work?

Consider the squence pa-ta-ka or (pha-tha-kha for plosive sounds), there should be a slience between two sounds.

<img width="942" alt="image" src="https://user-images.githubusercontent.com/112830629/188341304-3cd461ae-8cad-49e6-8733-f2a28f7f9802.png">


**Vocal folds do not vibrate in sounds[p]/[t]/[k]**

**Actually the task is done if the .wav file only contains [p]/[t]/[k] by setting appropriate threshold in `split_on_silence()`. But I am thinking to make this more general**

### Step2: Create a dataset using the `.wav` files
In this case, there is only ptk pattern in the .wav file. Therefore I used `file % 3` to divide them into three classes.

**[k] is the class 0, [p] is the class 1, [t] is the class 2.**

.wav files are already be prepared in `\waves` 

### Step 3: Data Augmentation
I shifted and stretched the .wav files to extend my dataset.

### Step 4: Tranfer .wav files into Spectrograms
.wav files are already be prepared in `\waves_img` 

### Step 5: Train the CNN model
Trained models are in `\waves_img\model`

### Step 6: Prepare the wav file as the same way as Step1 and Step4.

### Step 7: Use the best model from `\waves_img\model` to do the prediction 

**not yet done**

This should return a list of predictions: ['p', 't', 'k', ...] 

### Step 8: Count the pattern ptk by iterating the list

## Dependencies
Run the code:

```
pip install -r requirements.txt
```

## Train the model
```
python ptk.py
```
It should return something like this:
<img width="802" alt="image" src="https://user-images.githubusercontent.com/112830629/188346052-51da9083-2fe1-41e5-9756-3db5fdd5384b.png">


## Get the ptk count
```
python pred.py
```

You can use any model in `waves_img\model` here I have used the latest model.
