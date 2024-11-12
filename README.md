# Combining Deep Learning Intuition and Domain-Specific Language in the ARC Challenge 2024

The [Abstraction and Reasoning Corpus (ARC)](https://arcprize.org/) is a difficult benchmark due to its resistance to memorization. Deep Learning approaches that directly map input grids to output grids become increasingly ineffective as the private test set introduces novel tasks that deviate from known patterns.
Our contributions combine two key components. First, we develop an autoencoder model to generate abstract representations of individual grids, which are then aggregated into what we call "intuition vectors" of tasks. Through UMAP visualization, we demonstrate that these vectors capture meaningful semantic relationships between similar tasks, providing direction, or intuition for potential solutions. Second, we extend the established Domain-Specific Language (DSL) approach based on ARC's core knowledge priors, by introducing a Last-In-First-Out (LIFO) memory mechanism which we argue is fundamental for sequential reasoning by allowing an agent to store and retrieve intermediate states.
We propose that solving ARC lies in a combination of collective human effort to generate different DSL paths, or solutions for a same task and using the abstract representations to guide a future deep learning approach to extend the current dictionary of solutions to new unseen cases.

<div align="center">
<img src="https://github.com/user-attachments/assets/f04c9a97-64f9-4c19-aa79-8e1939336f52" width="65%">
</div>

# Setup

## Web interface

Inside of the `/web` folder, run :

```
npm install
npm run dev
```

## Flask backend

To run the Flask server :

- make sure you have `\app\api\env\intuition_models\deep_arc_autoencoder_256.pth` file
- make sure you have the [ARC dataset file](https://github.com/fchollet/ARC/archive/master.zip), and place it here: `\app\api\env\ARC-AGI-master.zip`
- (optional) if deploying to a production environment, update session key in `config.py` file

Activate Python virtualenv at the root of project :

```
py -3 -m venv .venv
.venv\Scripts\activate
```

Install requirements :

```
pip install -r requirements.txt
```

Run the Flask server in debug mode :

```
flask --app run --debug run
```
