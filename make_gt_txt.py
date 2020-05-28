import os
import sys
import argparse
import fire
import numpy as np

# 파일을 순회하며 현재 dir 위치/ 파일명 + 확장자명 을 txt 파일에 입력한다.
#필요한 parameters : 쓸 txt 파일 명 / 순회할 dir 경로 /

def make_gt_txt(file_name, dir_path, select_data ='/') :
    """ select_data = '/' contains all sub-directory of root directory """
    os.chdir(dir_path) # C:\Users\green\Desktop\MAL_LP~1\Mal_LP_master\번호판이미지들\Image_Rotated
    with open(file_name, mode='wt', encoding = 'utf-8') as gt:

        for dirpath, cont_dirs, cont_files in os.walk(dir_path):
            # print('\niter:', i)
            i = 1
            # print("a 현재 주소 :",a,'\nb속한 폴더 목록 :',b,'\nc속한 파일 목록 :',c)
            if not cont_dirs:
                print(dirpath)
                cont_files.sort()
                for file in cont_files:
                    write_f = os.path.join(dirpath, file)+'\t\n'
                    print(write_f)
                    gt.write(write_f)
                    i += 1
        print( i )


#1. f.open 을 한다.

#2. 입력받은 dir 을 하위 dir까지 순회하며 파일명을 반환한다.


#3. 파일에 한 줄씩 쓴다. 이때, 파일명 + \t 하고 \n 해야함.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name',required = True)
    parser.add_argument('--dir_path', required = True)
    opt = parser.parse_args()

    make_gt_txt(opt.file_name,opt.dir_path)



#USAGE:"python make_gt_txt.py --file_name gt.txt --dir_path /home/pirl/PycharmProjects/untitled2/plate_recognition/Image_Rotated/"


