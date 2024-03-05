import time


# label file processing
txtFile_paths = []

with open(file="check_exist_or_not/train.txt", mode='r') as txtFiles:
    lines = txtFiles.readlines()

    for line in lines:
        txtFile_paths.append(line.split('/')[-1].split('.')[0].strip())


# image file processing
jpgFile_paths = []

with open(file="check_exist_or_not/train_label.txt", mode='r') as jpgFiles:
    lines = jpgFiles.readlines()

    for line in lines:
        jpgFile_paths.append(line.split('/')[-1].split('.')[0].strip())


new_jpgFile_paths = []
# 13.031832933425903
for txt in txtFile_paths:
    if txt in jpgFile_paths:
        new_jpgFile_paths.append(f"/home/gpuadmin/ZPD/Bdd_uncleaned/3class_bdd/train/" + txt + ".jpg")
    else:
        # print(txt)
        pass

print(len(new_jpgFile_paths))

with open(file=f"check_exist_or_not/train{len(new_jpgFile_paths)}.txt", mode='w') as new_jpgFiles:
    for paths in new_jpgFile_paths:
        new_jpgFiles.write(paths + "\n")

