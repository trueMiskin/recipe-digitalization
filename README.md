# Installing dependecies

`pip install -r requirements.txt`

There is a bug in `paddleocr` library. You can download fix version from this repository:

```
git clone --depth 1 git@github.com:trueMiskin/PaddleOCR.git
cd PaddleOCR
pip install -r requirements.txt
pip install .
```