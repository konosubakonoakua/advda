import json
import os
from typing import List

def get_imagenet_classes() -> dict:
    # print(os.path.abspath("."))
    # print(os.path.curdir)
    # 坑壁啊，居然默认是在整个项目的根目录下
    with open("./utils/imagenet_classes.json", 'r') as f:
        mapping = json.loads(f.read())
    return mapping

def get_imagenet_classes_zh() -> List[str]:
    # print(os.path.abspath("."))
    # print(os.path.curdir)
    # 坑壁啊，居然默认是在整个项目的根目录下
    with open("./utils/imagenet_classes_zh.json", 'r', encoding='utf8') as f:
        mapping = json.loads(f.read())
    return mapping