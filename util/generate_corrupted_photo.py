from PIL import Image, ImageFont, ImageDraw
import os
import cv2
import numpy as np

mask_list = os.listdir('./mask')
file_list = os.listdir('../data')


#

def generate_mesh_face(mesh_path, img_path, output_dir, mesh_type):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print(
            'create directory :', output_dir
        )
    mesh = cv2.imread(mesh_path)
    mesh = cv2.resize(mesh, (178, 220))
    face = cv2.imread('../data/' + img_path)
    face = cv2.resize(face, (178, 220))
    meshgray = cv2.cvtColor(mesh, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(meshgray, 150, 255, cv2.THRESH_BINARY)
    mesh_face = cv2.bitwise_and(face, face, mask=mask)
    f_name = img_path.split('.')[0] + 'mesh_%d.jpg' % mesh_type
    # print(img_path)
    # print (f_name)
    mesh_face_path = os.path.join(output_dir, f_name)
    # print(mesh_face_path)
    cv2.imwrite(mesh_face_path, mesh_face)
    # print('write image '+ f_name)
    return mesh_face


# dir for images
train_dir = '../train_mesh_data'
val_dir = '../val_mesh_data'

# generate mesh face
idx = 0
image_idx = 0
for file_path in file_list[0:-50]:
    train_num = (len(file_list) - 50) * (len(mask_list) - 1)

    for mesh_type in range(1, len(mask_list)):
        # print(mesh_type)

        maskPath = 'mask/' + mask_list[mesh_type]

        generate_mesh_face(maskPath, file_path, train_dir, mesh_type)
        idx += 1
        if idx % 100 == 0:
            print('%d train files done!' % idx)

idx = 0
for file_path in file_list[-50:]:
    maskPath = 'mask/' + mask_list[0]
    generate_mesh_face(maskPath, file_path, val_dir, 0)
    idx += 1
    if idx % 100 == 0:
        print('%d val files done!' % idx)

# mesh = cv2.imread('./mask/'+ mask_list[mesh_type])
# mesh = cv2.resize(mesh,(178,220))
# print('mesh shape ',mesh.shape)
# face = cv2.imread('../data/'+file_list[2])
# print(file_list[2].split('.')[0])
# face = cv2.resize(face,(178,220))
# #target = cv2.addWeighted(mask,0.5,face,0.8,0)
#
#
# meshgray = cv2.cvtColor(mesh, cv2.COLOR_BGR2GRAY)
# ret,mask = cv2.threshold(meshgray,150,255,cv2.THRESH_BINARY)
# mask_inv = cv2.bitwise_not(mask)
# mesh = cv2.bitwise_and(mesh,mesh,mask= mask_inv)
#
#
# print('mask shape ', mask.shape)
#
# mesh_face = cv2.bitwise_and(face,face,mask = mask)
# print('mesh_face shape ', mesh_face.shape)
# cv2.imwrite('../train_mesh_data/'+file_list[2].split('.')[0]+'mesh1.jpg',mesh_face)
#
# t = cv2.add(mesh_face , mesh)
# cv2.imshow('a',mesh_face)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
