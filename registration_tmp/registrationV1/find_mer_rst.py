import os
import cv2
import shutil
from random import random
def show_mer_rst(new_merged_path,find_mer_rst_path):
    _files = []
    rootdir = new_merged_path
    list_file = os.listdir(rootdir)
    print(list_file)

    e=0
    for i in range(0, len(list_file)):
        
        # 构造路径
        err=list_file[i]
        print('dicomi:',err)
        dicomi = os.path.join(rootdir,err )
        find_mer_rst_path_each_dicom=os.path.join(find_mer_rst_path,err)
        if not os.path.exists(find_mer_rst_path_each_dicom):
            os.makedirs(find_mer_rst_path_each_dicom)

        framesj = os.listdir(dicomi)
        format_file = ''
        num_dict={}
        try:
            for j2 in range(0, len(framesj)):
                # if os.path.isdir(framesj[j2]):
                #     with open(os.path.join(framesj[j2],os.path.basename(format_file)),'w') as fw:
                #         fw.write(format_file)
                if framesj[j2].endswith('.dcm') or framesj[j2].endswith('.jpg'):
                    continue
                else:
                    in_frames_j=os.path.join(dicomi,framesj[j2])
                    in_frames_j_res=os.path.join(in_frames_j,'res')

                    mer_frames_file=os.listdir(in_frames_j_res)[0]
                    img = cv2.imread(os.path.join(in_frames_j_res,mer_frames_file))
                    cv2.imwrite(os.path.join(find_mer_rst_path_each_dicom,mer_frames_file), img)
        except:
            e+=1
    print('e:',e)

if __name__=='__main__':
    mer_rst_path='/home/DataBase4/cto_gan_data/RAO_CAU/merged'
    # find_mer_rst_path='/home/DataBase4/cto_gan_data/RAO_CAU/find_merged_result'
    # if not os.path.exists(find_mer_rst_path):
    #     os.makedirs(find_mer_rst_path)
    # show_mer_rst(mer_rst_path,find_mer_rst_path)

    '''plot_mer_rst.py'''


    '''person choose paires index'''
    #其实可以多选择
    with open('/home/DataBase4/cto_gan_data/RAO_CAU/case100.csv') as fr:
        A_path='/home/DataBase4/cto_gan_data/RAO_CAU/merged_paires/A'
        B_path='/home/DataBase4/cto_gan_data/RAO_CAU/merged_paires/B'
        for line in fr:
            line=line.strip('\n')
            li=line.split(',')
            dicom_index=li[0]
            dicom_path=os.path.join(mer_rst_path,'dicom'+str(dicom_index))
            new_li=[]
            for i in li:
                if i=='':
                    continue
                new_li.append(i)
            format_f=new_li[-1]
            for pair in new_li[1:-1]:
                frames_paires_path=os.path.join(dicom_path,'frames'+str(pair),'moved')
                png_format_path=''
                png_naive_path=''
                png_naive_name=''
                rand=random()
                if rand<=0.1:
                    train_or_test='test'
                else:
                    train_or_test='train'
                A=os.path.join(A_path,train_or_test)
                B=os.path.join(B_path,train_or_test)
                png_path=''
                for png in os.listdir(frames_paires_path):
                    png_index=png.split('_')[-1].split('.')[0]
                    png_path=os.path.join(frames_paires_path,png)

                    if not os.path.exists(A):
                        os.mkdir(A)
                    if not os.path.exists(B):
                        os.mkdir(B)
                    if png_index==format_f:
                        #long A
                        png_format_path=png_path

                    if png_index!=format_f:
                        #short
                        png_naive_path=png_path
                        png_naive_name=png
                try:

                    shutil.copy(png_format_path, A+'/'+png_naive_name)

                    shutil.copy(png_naive_path,B+'/'+png_naive_name)
                except:
                    print(frames_paires_path)

           
                    



    '''get paries moved rst to new path for pix2pix train'''
    
