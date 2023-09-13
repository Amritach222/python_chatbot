# Chatbot Deployment with Flask
## Initial Setup:

Clone repo and create a virtual environment
```
$ git clone https://github.com/Amritach222/python_chatbot.git
$ cd python_chatbot
$ python3 -m venv venv
$  venv/Scripts/activate
```
Install dependencies
```
$ (venv) pip install Flask torch torchvision nltk
```
Install nltk package
```
$ (venv) python
>>> import nltk
>>> nltk.download('punkt')
```
Modify `intents.json` with different intents and responses for your Chatbot

Run
```
$ (venv) python train.py
```
This will dump data.pth file. And then run
the following command to test it in the console.
```
$ (venv) python chat.py
```

 To get response through API
 $ (venv) python app.py

