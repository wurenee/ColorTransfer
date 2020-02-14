#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np
import sys


# In[5]:


# These are the matrices used 
RGB_LMS = np.array([[0.3811,0.5783,0.0402],[0.1967,0.7244,0.0782],[0.0241,0.1288,0.8444]],dtype=np.float32)
LMS_LAB = np.matmul(np.array([[1/np.sqrt(3),0,0],[0,1/np.sqrt(6),0],[0,0,1/np.sqrt(2)]]),
                    np.array([[1,1,1],[1,1,-2],[1,-1,0]]))
LAB_LMS = np.matmul(np.array([[1,1,1],[1,1,-1],[1,-2,0]]),
                    np.array([[1/np.sqrt(3),0,0],[0,1/np.sqrt(6),0],[0,0,1/np.sqrt(2)]]))
LMS_RGB = np.array([[4.4679, -3.5873,0.1193],[-1.2186,2.3809,-0.1624],[0.0497,-0.2439,1.2045]])
LMS_CIE = np.array([[2,1,0.05],[1,-1.09,0.09],[0.11,0.11,-0.22]])
CIE_LMS = np.linalg.inv(LMS_CIE)


# In[ ]:





# In[6]:


def convert_color_space_BGR_to_RGB(img_BGR):
    img_RGB = np.zeros_like(img_BGR,dtype=np.float32)
    img_RGB = img_BGR[:,:,::-1]
    return img_RGB


# In[7]:


def convert_color_space_RGB_to_BGR(img_RGB):
    img_BGR = np.zeros_like(img_RGB,dtype=np.float32)
    img_BGR = img_RGB[:,:,::-1]
    return img_BGR


# In[26]:


def convert_color_space_RGB_to_Lab(img_RGB):
    '''
    convert image color space RGB to Lab
    '''
    # Convert RGB to LMS
    img_LMS = np.zeros_like(img_RGB,dtype=np.float32)
    for i in range(img_RGB.shape[0]):
        for j in range(img_RGB.shape[1]):
            img_LMS[i,j,:] = np.matmul(RGB_LMS,img_RGB[i,j,:])
    img_LMS = np.log10(img_LMS)
    #Convert LMS to Lab
    img_Lab = np.zeros_like(img_RGB,dtype=np.float32)
    for i in range(img_RGB.shape[0]):
        for j in range(img_RGB.shape[1]):
            img_Lab[i,j,:] = np.matmul(LMS_LAB,img_LMS[i,j,:])
    return img_Lab


# In[9]:


def convert_color_space_Lab_to_RGB(img_Lab):
    '''
    convert image color space Lab to RGB
    '''
    img_LMS = np.zeros_like(img_Lab,dtype=np.float32)
    # The matrix for transforming LAB into LMS before power to the 10th

    for i in range(img_Lab.shape[0]):
        for j in range(img_Lab.shape[1]):
            img_LMS[i,j,:] = np.matmul(LAB_LMS,img_Lab[i,j,:])
    img_LMS = np.power(10,img_LMS)
    
    img_RGB = np.zeros_like(img_Lab,dtype=np.float32)

    
    
    for i in range(img_Lab.shape[0]):
        for j in range(img_Lab.shape[1]):
            img_RGB[i,j,:] = np.matmul(LMS_RGB,img_LMS[i,j,:])

    return img_RGB


# In[10]:


def convert_color_space_RGB_to_CIECAM97s(img_RGB):
    '''
    convert image color space RGB to CIECAM97s
    '''
    img_CIECAM97s = np.zeros_like(img_RGB,dtype=np.float32)
    
    #First find the LMS 
    img_LMS = np.zeros_like(img_RGB,dtype=np.float32)
    for i in range(img_RGB.shape[0]):
        for j in range(img_RGB.shape[1]):
            img_LMS[i,j,:] = np.matmul(RGB_LMS,img_RGB[i,j,:])
            img_CIECAM97s[i,j,:] = np.matmul(LMS_CIE,img_LMS[i,j,:])
    return img_CIECAM97s


# In[11]:


def convert_color_space_CIECAM97s_to_RGB(img_CIECAM97s):
    '''
    convert image color space CIECAM97s to RGB
    '''
    img_RGB = np.zeros_like(img_CIECAM97s,dtype=np.float32)
    img_LMS = np.zeros_like(img_RGB,dtype=np.float32)
    for i in range(img_CIECAM97s.shape[0]):
        for j in range(img_CIECAM97s.shape[1]):
            img_LMS[i,j,:] = np.matmul(CIE_LMS,img_CIECAM97s[i,j,:])
            img_RGB[i,j,:] = np.matmul(LMS_RGB,img_LMS[i,j,:])
    return img_RGB


# In[12]:


def color_transfer_in_Lab(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_Lab =====')
    # convert RGB into Lab
    img_Lab_source = np.zeros_like(img_RGB_source,dtype=np.float32)
    img_Lab_target = np.zeros_like(img_RGB_target,dtype=np.float32)
    img_Lab_source = convert_color_space_RGB_to_Lab(img_RGB_source)
    img_Lab_target = convert_color_space_RGB_to_Lab(img_RGB_target)
    # find the l,a,b to find mean and standard deviation
    l_source = img_Lab_source[:,:,0]
    l_target = img_Lab_target[:,:,0]
    a_source = img_Lab_source[:,:,1]
    a_target = img_Lab_target[:,:,1]
    b_source = img_Lab_source[:,:,2]
    b_target = img_Lab_target[:,:,2]
    l_final = ((l_source - np.mean(l_source))* (np.std(l_target)) / (np.std(l_source))) + np.mean(l_target)
    a_final = ((a_source - np.mean(a_source))* (np.std(a_target)) / (np.std(a_source))) + np.mean(a_target)
    b_final = ((b_source - np.mean(b_source))* (np.std(b_target)) / (np.std(b_source))) + np.mean(b_target)
    img_Lab_result = np.dstack((l_final,a_final,b_final))
    img_RGB_result = convert_color_space_Lab_to_RGB(img_Lab_result)
    return img_RGB_result


# In[13]:


def color_transfer_in_RGB(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_RGB =====')
    l_source = img_RGB_source[:,:,0]
    l_target = img_RGB_target[:,:,0]
    a_source = img_RGB_source[:,:,1]
    a_target = img_RGB_target[:,:,1]
    b_source = img_RGB_source[:,:,2]
    b_target = img_RGB_target[:,:,2]
    l_final = ((l_source - np.average(l_source))* (np.std(l_target)) / (np.std(l_source))) + np.average(l_target)
    a_final = ((a_source - np.average(a_source))* (np.std(a_target)) / (np.std(a_source))) + np.average(a_target)
    b_final = ((b_source - np.average(b_source))* (np.std(b_target)) / (np.std(b_source))) + np.average(b_target)
    img_RGB_result = np.dstack((l_final,a_final,b_final))
    return img_RGB_result


# In[14]:


def color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_CIECAM97s =====')
    img_CIECAM97s_source = convert_color_space_RGB_to_CIECAM97s(img_RGB_source)
    img_CIECAM97s_target = convert_color_space_RGB_to_CIECAM97s(img_RGB_target)
    l_source = img_CIECAM97s_source[:,:,0]
    l_target = img_CIECAM97s_target[:,:,0]
    a_source = img_CIECAM97s_source[:,:,1]
    a_target = img_CIECAM97s_target[:,:,1]
    b_source = img_CIECAM97s_source[:,:,2]
    b_target = img_CIECAM97s_target[:,:,2]
    l_final = ((l_source - np.average(l_source))* (np.std(l_target)) / (np.std(l_source))) + np.average(l_target)
    a_final = ((a_source - np.average(a_source))* (np.std(a_target)) / (np.std(a_source))) + np.average(a_target)
    b_final = ((b_source - np.average(b_source))* (np.std(b_target)) / (np.std(b_source))) + np.average(b_target)
    img_CIECAM97s_result = np.dstack((l_final,a_final,b_final))
    img_RGB_result = convert_color_space_CIECAM97s_to_RGB(img_CIECAM97s_result)
    return img_RGB_result


# In[15]:


def color_transfer(img_RGB_source, img_RGB_target, option):
    if option == 'in_RGB':
        img_RGB_new = color_transfer_in_RGB(img_RGB_source, img_RGB_target)
    elif option == 'in_Lab':
        img_RGB_new = color_transfer_in_Lab(img_RGB_source, img_RGB_target)
    elif option == 'in_CIECAM97s':
        img_RGB_new = color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target)
    return img_RGB_new


# In[16]:


if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, Winter 2020, HW1: color transfer')
    print('==================================================')

    path_file_image_source = sys.argv[1]
    path_file_image_target = sys.argv[2]
    path_file_image_result_in_Lab = sys.argv[3]
    path_file_image_result_in_RGB = sys.argv[4]
    path_file_image_result_in_CIECAM97s = sys.argv[5]

    # ===== read input images
    
    img_RGB_source = cv2.imread(path_file_image_source).astype(np.float32)/255.0
    img_RGB_target = cv2.imread(path_file_image_target).astype(np.float32)/255.0

    img_RGB_new_Lab       = color_transfer(img_RGB_source, img_RGB_target, option='in_Lab')
    cv2.imwrite(path_file_image_result_in_Lab,np.clip(img_RGB_new_Lab*255, 0.0, 255.0).astype(np.uint8))
    
    img_RGB_new_RGB       = color_transfer(img_RGB_source, img_RGB_target, option='in_RGB')
    #cv2.imwrite(path_file_image_result_in_RGB,img_RGB_new_RGB*255.clip(0,255.0).astype(np.uint8))
    cv2.imwrite(path_file_image_result_in_RGB,np.clip(img_RGB_new_RGB*255, 0.0, 255.0).astype(np.uint8))

    img_RGB_new_CIECAM97s = color_transfer(img_RGB_source, img_RGB_target, option='in_CIECAM97s')
    cv2.imwrite(path_file_image_result_in_CIECAM97s,np.clip(img_RGB_new_CIECAM97s*255, 0.0, 255.0).astype(np.uint8))


# In[30]:




