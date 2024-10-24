# ARC

Abstraction and Reasoning Corpus (ARC) solver attempt.

# Flask server

To run the Flask server :

- make sure you have `\app\api\env\intuition_models\deep_arc_autoencoder_256.pth` file
- make sure you have `\app\api\env\ARC-AGI-master.zip` dataset file

- activate Python virtualenv at the root of project :

```
py -3 -m venv .venv
.venv\Scripts\activate
```

- install requirements :

```
pip install -r requirements.txt
```

- run the Flask server in debug mode :

```
flask --app run --debug run
```
