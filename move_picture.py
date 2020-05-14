import os
import shutil

nof = open("noGlass.txt")
hasf = open("Glass.txt")

noLine = nof.readline()
hasLine = hasf.readline()

list = os.listdir("./img_align_celeba")
hasGo = True
noGo = True
for i in range(0, len(list)):
    imgName = os.path.basename(list[i])
    if (os.path.splitext(imgName)[1] != ".jpg"): continue

    noArray = noLine.split()
    if (len(noArray) < 1): noGo = False
    hasArray = hasLine.split()
    if (len(hasArray) < 1): hasGo = False

    if (noGo and (imgName == noArray[0])):
        oldname= "./img_align_celeba/"+imgName
        newname="./noGlass/"+imgName
        shutil.move(oldname, newname)
        noLine = nof.readline()

    if (hasGo and (imgName == hasArray[0])):
        oldname= "./img_align_celeba/"+imgName
        newname="./wearGlass/"+imgName
        shutil.move(oldname, newname)
        hasLine = hasf.readline()

    if (i % 100 == 0): print(imgName)
print("wear glasses move done!")

for i in range(0, len(list)):
    imgName = os.path.basename(list[i])
    oldname = "./img_align_celeba/" + imgName
    newname = "./noGlass/" + imgName
    shutil.move(oldname, newname)
    noLine = nof.readline()
    print(imgName)
print("move done!")

nof.close()
hasf.close()