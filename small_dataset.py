import os

list1 = os.listdir("./noGlass")
list2 = os.listdir("./wearGlass")

for i in range(0, len(list1)):
    if i > 4000:
        imgName = os.path.basename(list1[i])
        newname = "./noGlass/" + imgName
        os.remove(newname);
        print(imgName, 'in noGlass has been removed')

for i in range(0, len(list2)):
    if i > 4000:
        imgName = os.path.basename(list2[i])
        newname = "./wearGlass/" + imgName
        os.remove(newname);
        print(imgName, 'in wearGlass has been removed')

print("remove done!")