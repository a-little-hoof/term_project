#查看和显示nii.gz文件
 
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
from PIL import Image
 
example_filename = './img0067.nii.gz'
img = nib.load(example_filename)
print(img)


width, height, queue = img.dataobj.shape
#OrthoSlicer3D(img.dataobj).show()
print(queue) 
num = 1
for i in range(0, queue, 10):
    img_arr = img.dataobj[:, :, i]
    name = "img"+str(i)+".png"
    im = Image.fromarray(img_arr)
    im.save(name)
 
 