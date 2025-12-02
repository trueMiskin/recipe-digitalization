# Installing dependecies

`pip install -r requirements.txt`

There is a bug in `paddleocr` library. You can download fix version from this repository:

```
git clone --depth 1 git@github.com:trueMiskin/PaddleOCR.git
cd PaddleOCR
pip install -r requirements.txt
pip install .
```

First run, `dataset_creator.py` and then `dataset_crator2.py`.
After that you can train the model by `python model2.py`.

Testing model/generate predictions can be done by running `python model2.py infer ...`