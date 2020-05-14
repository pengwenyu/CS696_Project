f = open("list_attr_celeba.txt")
newTxt = "Glass.txt"
new = open(newTxt, "a+")
newNoTxt = "noGlass.txt"
newNof = open(newNoTxt, "a+")

line = f.readline()
line = f.readline()
line = f.readline()
while line:
    array = line.split()
    if array[0] == "000053.jpg":
        print(array[16])

    if array[16] == "-1":
        new_context = array[0] + '\n'
        newNof.write(new_context)
    else:
        new_context = array[0] + '\n'
        new.write(new_context)
    line = f.readline()

count=0
thefile =open("noGlass.txt")
while True:
    buffer=thefile.read(2048*8192)
    if not buffer:
        break
    count += buffer.count('\n')
thefile.close()
print('There are', count, 'lines in noGlass')

count=0
thefile =open("Glass.txt")
while True:
    buffer=thefile.read(2048*8192)
    if not buffer:
        break
    count += buffer.count('\n')
thefile.close()
print('There are', count, 'lines in Glass')


f.close()
new.close()
newNof.close()
