
def get_dir(dir_txt):
    with open(dir_txt ,"r") as f:
        file_list =[file for line in f for file in line.split()]
    return file_list

def ret_dir(file_loc ,file_list):
    dir_root = str(file_loc)
    conv_ = list()

    for file in file_list:
        dir_= dir_root +file
        conv_ += [dir_]

    with open('dir_list.txt', 'w+') as f:
        for dir_ in conv_:
            f.write("%s\n" % dir_)


def main():
    dir_txt = input("name of text file that contain list of image files: ")
    file_loc = input("directory where your image files saved: ")
    file_list = get_dir(dir_txt)
    ret_dir(file_loc ,file_list)
    print("[ Check 'dir_list.txt' in the same folder] ")

if __name__== "__main__":
    main()