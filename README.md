# open-ended-question-eval


### Intro
This is my capstone project at UT-Austin MSIS 2023.
The open-ended question is a widely used method in human-subject surveys. Open-ended questions provide opportunities for discovering human subjects’ spontaneous responses to the questions. However, it is labor-intensive and time- consuming to analyze the responses.
In light of the challenge, I aim to leverage NLP methods to create a tool for assisting the analysis of open-ended questions.

##### Code Extraction
I extracted the codes using [PyABSA](https://github.com/yangheng95/PyABSA).
The sample data is also aquire from the repo.

##### Code Classification
I implemented the classifier leveraging GPT-neo released by EleutherAI. The model can be accessed on  [Hugging Face](https://huggingface.co/EleutherAI/gpt-neo-1.3B)

### Workflow

<img src="https://i.imgur.com/bDa2gTf.png" height="50%" width="50%">

### Usage

1. Packages

```
pip install -r requirement.txt
```

2. Codes Extraction

```
python main.py --file_path <path to responses file> --extraction

```
The codes will be saved as `codes.csv`. You can edit the codes before you do classification.

3. Classification

```
python main.py --file_path <path to responses file> --classification
```
The results will saved as json files.

You can also run `python main.py --file_path <path to responses file> --extraction --classification`. However, you will not be able to edit the codes.




