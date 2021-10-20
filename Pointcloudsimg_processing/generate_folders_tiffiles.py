import os
import shutil

def clipped_images(tif_dire,father_dir):
    tifiles = os.listdir(tif_dire)
    for tif_file in tifiles:
        if tif_file[-4:] == ".tif":
            tifimage_path = father_dir+"\\"+tif_file[:-4]
    print("True")

    return True

def generate_folders(tif_dire,father_dir):
    tifiles = os.listdir(tif_dire)
    for tif_file in tifiles:
        if tif_file[-4:] == ".tif":
            print("True")
            os.makedirs(father_dir+"\\"+tif_file[:-4]+"\\"+"Tifimages")
            des_folder= father_dir+"\\"+tif_file[:-4]+"\\"+"Tifimages"
            shutil.copy2(tif_dire+"\\"+tif_file,des_folder)

    return True

if __name__ == '__main__':
    tif_dire = "E:\\Point_clouds_processing_yan\\Areas_Pointclouds\\Area3_Terencegdb\\Tifimages"
    father_dir = "E:\\Point_clouds_processing_yan\\Areas_Pointclouds\\Area3_Terencegdb"
    # generate_folders(tif_dire, father_dir)
    clipped_images(tif_dire, father_dir)
    print("True")
