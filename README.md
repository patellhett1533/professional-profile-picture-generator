# Professional Profile Picture Generator

A python project which generate a professional profile pictures from your given images using stable diffusion

for setup

```javascript
git clone git@github.com:patellhett1533/professional-profile-picture-generator.git
```

then create virtual environment

```javascript
python3 -m venv venv
```

then activate them

```javascript
source venv/bin/activate
```

then install all packages

```javascript
pip install -r requirements.txt
```

after that create .env file and set default value

```javascript
PROJECT_NAME = "proile_picture_generator";
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0";
LEARNING_RATE = "1e-4";
NUM_STEPS = "500";
BATCH_SIZE = "1";
GRADIENT_ACCUMULATION = "4";
RESOLUTION = "1024";
USE_8BIT_ADAM = "False";
USE_XFORMERS = "False";
MIXED_PRECISION = "fp16";
TRAIN_TEXT_ENCODER = "False";
DISABLE_GRADIENT_CHECKPOINTING = "False";
```

then create a folder in root directory named "images" and put your normal image inside this folder.

now run the generate.py file and make sure your venv is active

```javascript
python generate.py
```

it will take around 2 hours for generating a profile pictures and and save into images folder.
