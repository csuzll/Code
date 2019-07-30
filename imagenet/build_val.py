"""将ImageNet中的val数据按wordId目录方式存放，并将val黑名单中的文件存储为txt文件保留"""
from pathlib import Path
import os,shutil
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

# 定义验证集数据路径以及对应的标签文件路径
VAL_LIST = IMAGES_SETS_PATH / "val.txt"
VAL_LABELS = DEVKIT_PATH / "ILSVRC2015_clsloc_validation_ground_truth.txt"
# 定义val blacklisted 文件路径
VAL_BLACKLIST = DEVKIT_PATH / "ILSVRC2015_clsloc_validation_blacklist.txt"


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
        labelMappings[wordId] = int(label)
        
    return labelMappings

"""处理val数据集"""
def buildValidationSet(vallist_path, vallabels_path, valblacklist_path, image_path, labelMappings):

    # 验证数据集
    # ILSVRC2012_val_00000001 1(name, imageNum)
    valFilenames = open(str(vallist_path)).read().strip().split("\n")

    # 验证集对应的标签
    # 490(label)
    valLabels = open(str(vallabels_path)).read().strip().split("\n")
    
    # 黑名单数据
    # 36(imageNum)
    valBlacklist = set(open(str(valblacklist_path)).read().strip().split("\n"))
    
    # 在val路径下创建wordId目录
    wordIdlist = list(labelMappings.keys())
    labellist = list(labelMappings.values())
    val_path = image_path / "val"
    
    for wordId in wordIdlist:
        p = val_path / wordId
        p.mkdir(parents=True, exist_ok=True)

    # 移动val图片到相应的目录下
    for i, (row,label) in enumerate(zip(valFilenames, valLabels)):
        (partialPath, imageNum) = row.strip().split(" ")

        # 黑名单过滤
        if imageNum in valBlacklist:
            continue

        partialPathName = partialPath + ".JPEG"
        wordIddir = wordIdlist[labellist.index(int(label))]

        src_path = val_path / partialPathName
        target_dir = val_path / wordIddir 
        
        # 移动文件
        shutil.move(str(src_path), str(target_dir))


# 保存为txt
def text_save(filename, data):
    file = open(filename, "w+")
    for i in range(len(data)):
        s = data[i].replace("'",'') + "\n"
        file.write(s)
    file.close()
    print("保存文件成功")

# 主程序
if __name__ == "__main__":
    labelMappings = buildClassLabels(WORD_IDS)
    buildValidationSet(VAL_LIST, VAL_LABELS, VAL_BLACKLIST, IMAGES_PATH, labelMappings)
    
    # 将val中黑名单指示的图片文件名存储为txt
    val_path = IMAGES_PATH / "val"
    black_filenames = list(val_path.glob("*.JPEG"))
    black_names = [f.name for f in black_filenames]
    # 保存
    text_save("ILSVRC2015_clsloc_validation_blackfilename.txt", black_names)
    
    # 删除文件
    rmdir = val_path / "remove"
    rmdir.mkdir(parents=True, exist_ok=True)
    for p in black_filenames:
        shutil.move(str(p), str(rmdir))
        

# 然后在服务器上手动rm -rf remove