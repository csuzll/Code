from pathlib import Path
import os,shutil
import numpy as np
from sklearn.model_selection import train_test_split


# 定义Imagenet数据集路径
BASE_PATH = Path("ILSVRC2017")

# 基于base_path定义原始图像和工具路径
IMAGES_PATH = BASE_PATH / "Data" / "CLS-LOC" # 包含train,val和test数据集
IMAGES_SETS_PATH = BASE_PATH / "ImageSets" / "CLS-LOC" # 包含重要的train_cls.txt和val.txt
DEVKIT_PATH = BASE_PATH / "devkit" / "data" # devkit所在位置的基本路径

# 定义WordNet IDs文件路径
WORD_IDS = DEVKIT_PATH / "map_clsloc.txt"

# 定义training文件路径
TRAIN_LIST = IMAGES_SETS_PATH / "train_cls.txt"

# 定义类别个数
# 定义我们需要从train数据集中划分一个子集作为test数据集
NUM_CLASSES = 1000
NUM_TEST_IMAGES = 50 * NUM_CLASSES

"""构建类标签映射字典"""
def buildClassLabels(wordid_path):
    # 文件名映射类标签
    # n02110185 3 Siberian_husky
    """
    读取WORD_IDs文件的全部内容，并构建一个labelMappings字典，其中key为WordNet ID，值为整数类标签。
    """
    rows = open(str(wordid_path)).read().strip().split("\n")
    labelMappings = {}

    # 遍历整个文件内容，对于每一行，我们分解为一个3元祖（wordId，label，实际可读标签)
    for row in rows:
        (wordId, label, hrLabel) = row.split(" ")
        labelMappings[wordId] = int(label)-1 # MATLAB编程语言是单索引的(即从1开始计数)，而Python编程语言是零索引的(从0开始计数)。
        
    return labelMappings

"""处理train数据集"""
def buildTrainingSet(trainlist_path, image_path, labelMappings):
    # 训练数据集
    # n01440764/n01440764_12131 189
    rows = open(str(trainlist_path)).read().strip().split("\n")
    paths = []
    labels = []

    # 对原始数据进行遍历，构建完整的图像路径
    for i, row in enumerate(rows):
        # partialPath对应图像的文件名，比如：n01440764/n01440764_10026。
        # imageNum变量只是一个计数器——它在构建train数据集时没有任何用途，可以忽略。
        (partialPath, imageNum) = row.strip().split(" ")

        (partialPathDir, partialPathName) = partialPath.split("/")

        partialPathFullName = partialPathName + ".JPEG"

        # 原始图像数据路径
        path = image_path / "train" / partialPathDir / partialPathFullName

        # wordId
        wordId = partialPathDir
        label = labelMappings[wordId]

        paths.append(path)
        labels.append(label)
    return (np.array(paths), np.array(labels))

"""随机划分train，并将划分出的test数据移动到新建的test的文件"""
def divide_movefile(trainpaths, trainlabels,testsize):
    print('[INFO] constructing splits...')
    split = train_test_split(trainpaths, trainlabels, test_size=testsize, stratify=trainlabels, random_state=42)
    trainPaths,testPaths,trainLabels,testLabels = split
    
    print("[INFO] move files")
    for testPath in testPaths:
        wordId = testPath.parent.name
        
        target_dir = str(testPath.parent).replace("train", "test")
        if not os.path.exists(target_dir):
            Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        shutil.move(str(testPath), target_dir)
    print("[INFO] move file success")
    
# 主程序
if __name__ == "__main__":
    labelMappings = buildClassLabels(WORD_IDS)
    trainpaths, trainlabels = buildTrainingSet(TRAIN_LIST, IMAGES_PATH, labelMappings)
    divide_movefile(trainpaths, trainlabels, NUM_TEST_IMAGES)