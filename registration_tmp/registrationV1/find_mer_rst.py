import os
import cv2


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
    mer_rst_path='/home/wly/Documents/cto_frames/registration_tmp/registrationV1/test1118/merged'
    find_mer_rst_path='/home/wly/Documents/cto_frames/registration_tmp/registrationV1/test1118/find_merged_result'
    if not os.path.exists(find_mer_rst_path):
        os.makedirs(find_mer_rst_path)


    show_mer_rst(mer_rst_path,find_mer_rst_path)