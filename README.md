# Installing dependecies

`pip install -r requirements.txt`

Then due to bug in a `paddleclas` library, installation for Python 3.12 fails.
To fix that, you need to download a git repository of paddleclas:

`git clone git@github.com:PaddlePaddle/PaddleClas.git`

Update `requirements.txt` file in the `PaddleClas` directory to use `numpy==1.26.4` instead of `numpy==1.24.4`. Then build a library:

`pip install .`

Additionally, there is a bug in `paddleocr` library. You can download fix version from this repository:

```
git clone --depth 1 git@github.com:trueMiskin/PaddleOCR.git
cd PaddleOCR
pip install -r requirements.txt
pip install .
```