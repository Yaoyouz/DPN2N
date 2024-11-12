import numpy as np
import os
np.random.seed(1234)
folder_path = '/lichunlong/data_seismic_3D'
file_paths = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
for file_path in file_paths:
    file_name = file_path.split('.')[0]
    # if file_name =='Parihaka3D_1'  or file_name =='norm_canning3d_GDA94_UTM50s_p2' or file_name =='F3' or \
    #      file_name == 'Ichthys3D_1' or file_name =='OPUNAKE3D_1':
    #     print(file_name)
    #     file_path = os.path.join(folder_path,file_path)
    #     folder_tv = ''
    #     data_size = (192,192)
    #     data = np.load(file_path)
    #     dim1,dim2,dim3 = data.shape
    #     num1,num3 = dim1//data_size[0],dim3//data_size[1]
    #     data = data[:num1*data_size[0],:,:num3*data_size[1]]
    #
    #     for i in range(dim2):
    #         for j in range(num1):
    #             for z in range(num3):
    #                 if np.random.rand() < 0.75:
    #                     folder_tv = 'train'
    #                 else:
    #                     folder_tv = 'val'
    #                 folder_output_path = os.path.join('/lichunlong/data_2D_192', folder_tv)
    #                 data_ones = data[j*data_size[0]:(j+1)*data_size[0],i,z*data_size[1]:(z+1)*data_size[1]]
    #                 ouput_path = folder_output_path+'/'+file_name+'_'+str(j)+'_'+str(i)+'_'+str(z)+'.npy'
    #                 np.save(ouput_path,data_ones)
    if file_name == 'F3':
        folder_tv = 'val'
    else:
        folder_tv = 'train'
    data_size = (192, 192)
    file_path = os.path.join(folder_path, file_path)
    data = np.load(file_path)
    dim1,dim2,dim3 = data.shape
    num1,num3 = dim1//data_size[0],dim3//data_size[1]
    data = data[:num1*data_size[0],:,:num3*data_size[1]]
    for i in range(dim2):
        for j in range(num1):
            for z in range(num3):
                folder_output_path = os.path.join('/lichunlong/data_2D_192', folder_tv)
                data_ones = data[j*data_size[0]:(j+1)*data_size[0],i,z*data_size[1]:(z+1)*data_size[1]]
                ouput_path = folder_output_path+'/'+file_name+'_'+str(j)+'_'+str(i)+'_'+str(z)+'.npy'
                np.save(ouput_path,data_ones)

