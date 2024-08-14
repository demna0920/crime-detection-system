import glob, os


data_dir = os.path.dirname(os.path.abspath(__file__))

print(data_dir)

data_dir = 'data/obj'


percentage_calculate = 10;

file_train = open('data/train.txt', 'w')
file_test = open('data/test.txt', 'w')


count = 1
index_test = round(100 / percentage_calculate)
for pathAndFilename in glob.iglob(os.path.join(data_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    if count == index_test:
        count = 1
        file_test.write("data/obj" + "/" + title + '.jpg' + "\n")
    else:
        file_train.write("data/obj" + "/" + title + '.jpg' + "\n")
        count = count + 1
