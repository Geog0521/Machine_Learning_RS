# -*- coding: utf-8 -*-
import gdal
from osgeo import ogr, osr

# shp_path = "./mask_use.shp"
# driver = ogr.GetDriverByName("ESRI Shapefile")
# data_source = driver.CreateDataSource(shp_path)
# srs = osr.SpatialReference()
# srs.ImportFromEPSG(4326) #这是WGS84,想用什么自己去搜下对应的编码就行了
# layer = data_source.CreateLayer("polygon", srs, ogr.wkbPolygon)
# feature = ogr.Feature(layer.GetLayerDefn())
# wa = 116.122741699
# ha = 40.080871582
# wa1 = 116.122741699
# ha1 = 40.168762207
# wa2 = 116.251831055
# ha2 = 40.168762207
# wa3 = 116.251831055
# ha3 = 40.080871582
# wkt = "POLYGON((" + str(wa)+ " " +str(ha)+ "," + str(wa1) + " " + str(ha1) + "," + str(wa2)+ " " +str(ha2)+ "," + str(wa3)+ " " +str(ha3) + "))"
# point = ogr.CreateGeometryFromWkt(wkt)
# feature.SetGeometry(point)
# layer.CreateFeature(feature)
# feature = None
# data_source = None
import gdal
from osgeo import ogr, osr
import os
import numpy as np

def read_img(filename):
    dataset=gdal.Open(filename)

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize

    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)

    # del dataset
    return im_width,im_height,im_proj,im_geotrans,im_data,dataset

def get_mask(img_path,out_shp,proj=4326): #4326代表WGS84，其他的代码自己查，或者根据栅格应该可以获取到，没时间查了，你们自己优化一下
    im_width,im_height,im_proj,im_geotrans,im_data,dataset = read_img(img_path)
    xleft = im_geotrans[0]
    yleft = im_geotrans[3]
    xright = im_geotrans[0] + im_width*im_geotrans[1] + im_height*im_geotrans[2]
    yright = im_geotrans[3] + im_width*im_geotrans[4] + im_height*im_geotrans[5]
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.CreateDataSource(out_shp)
    #srs = osr.SpatialReference()
    #srs.ImportFromEPSG(proj)
    srs = osr.SpatialReference(wkt=dataset.GetProjection())#我在读栅格图的时候增加了输出dataset，这里就可以不用指定投影，实现全自动了，上面两行可以注释了，并且那个proj参数也可以去掉了，你们自己去掉吧
    layer = data_source.CreateLayer("polygon", srs, ogr.wkbPolygon)
    feature = ogr.Feature(layer.GetLayerDefn())
    wa = xleft
    ha = yright
    wa1 = xleft
    ha1 = yleft
    wa2 = xright
    ha2 = yleft
    wa3 = xright
    ha3 = yright
    wkt = "POLYGON((" + str(wa)+ " " +str(ha)+ "," + str(wa1) + " " + str(ha1) + "," + str(wa2)+ " " +str(ha2)+ "," + str(wa3)+ " " +str(ha3) + "))"
    point = ogr.CreateGeometryFromWkt(wkt)
    point.CloseRings()  #这个必须要有，不然创建出来的矢量面有问题
    feature.SetGeometry(point)
    layer.CreateFeature(feature)
    feature = None
    data_source = None


def batch_processing(des_directory,txt_file,destination_tif):
    des_tif_files = os.listdir(des_directory)
    list_tif = ["265.tif"]
    with open(txt_file, 'w') as f:  # 设置文件对象
        str_import_arcpy = 'import arcpy'
        f.writelines(str_import_arcpy+'\n')  # 将字符串写入文件中
        for desc_tif in des_tif_files:
            suffix = desc_tif[-4:]  # desc_tif[-7:]
            if suffix == ".tif": # and desc_tif in list_tif:  # "int.tif"
                # print(desc_tif)
                shpfile = des_directory + "/" + desc_tif[:-4] + "_Training" + "/" + desc_tif[:-4] + ".shp"
                tif_path = des_directory + "/" + desc_tif
                clipped_tif = shpfile[:-4]+"clipped.tif"
                get_mask(tif_path, shpfile)
                desc_str = "desc = arcpy.Describe(" + "'"+shpfile+"'"+ ")"+ "\n"
                f.writelines(desc_str)
                f.writelines('ext = desc.extent' + '\n')
                # get_mask(img_path, shp_path)
                # arcpy.Clip_management(
                #     "image.tif", "1952602.23 294196.279 1953546.23 296176.279",
                #     "clip.gdb/clip", "#", "#", "NONE")
                # envelop_str = shpfile.extent
                processing_str = "arcpy.Clip_management(" +"'"+ destination_tif+"'" + "," + "str(ext)" + ","+ "'" + clipped_tif +"'"+", '#', 'NONE')"
                # generate a batch processing script for clipping an destination image with .shp files
                f.writelines(processing_str + '\n')

    print("the script has been geneated successfully!")

def multispectralim_to_array(tif_file_path,channel_last=True):

    gdalDS = gdal.Open(tif_file_path, gdal.GA_ReadOnly)
    height = gdalDS.RasterYSize
    width =  gdalDS.RasterXSize
    bands = gdalDS.RasterCount

    for band in range(bands):
        band += 1
        srcband = gdalDS.GetRasterBand(band)
        dataraster = srcband.ReadAsArray(0,0,width,height).astype(np.uint16)
        if band==1:
            data = dataraster.reshape((height,width,1))
        else:
            data = np.append(data,dataraster.reshape((width,height)),axis=2)

    if channel_last:
        pass
    else:
        data = data.transpose(2,0,1)
    gdalDS = None

    return width,height,bands,data

def write_tif_data(input_data_arr,tifile_name,dataset_driver,rows_n,cols_n,data_type,band_count=1):
    gdal_tiff_driver = gdal.GetDriverByName("GTiff")
    target = gdal_tiff_driver.Create(tifile_name, xsize= cols_n, ysize= rows_n, bands=band_count, eType=data_type)
    im_geotrans = dataset_driver.GetGeoTransform()
    target.SetGeoTransform(im_geotrans)
    target.SetProjection(dataset_driver.GetProjection())  # set projection
    out_band = target.GetRasterBand(1)
    out_band.SetNoDataValue(-999)
    out_band.WriteArray(input_data_arr) ###
    out_band.FlushCache()
    out_band.ComputeBandStats(False)

    del gdal_tiff_driver,target

import rsgislib.vectorutils


def delete_noval_polygons(shpfile_path):

    query_name = 1
    shp_data = ogr.Open(shpfile_path)  # open the China.shp
    lyr = shp_data.GetLayer()

    lyr.SetAttributeFilter("PXLVAL = " + "'" + str(query_name) + "'")  # The field names should be in English

    print(lyr.GetFeatureCount())
    driver = shp_data.GetDriver()  # get driver from the original source
    # create a .shp file
    saved_shppath = shpfile_path[:-4]+"2.shp"
    out_ds = driver.CreateDataSource(saved_shppath)
    out_layer = out_ds.CopyLayer(lyr, "shp")
    del out_ds, out_layer,shp_data

    #delete the original shp
    os.remove(shpfile_path)

    #rename the shp
    shpfile_path = saved_shppath
    query_name = 1
    shp_data = ogr.Open(shpfile_path)  # open the China.shp
    lyr = shp_data.GetLayer()
    lyr.SetAttributeFilter("PXLVAL = " + "'" + str(query_name) + "'")  # The field names should be in English
    print(lyr.GetFeatureCount())
    driver = shp_data.GetDriver()  # get driver from the original source
    # create a .shp file
    saved_shppath = shpfile_path[:-5]+".shp"
    out_ds = driver.CreateDataSource(saved_shppath)
    out_layer = out_ds.CopyLayer(lyr, "shp")
    del out_ds, out_layer,shp_data
    #delete the original shp
    os.remove(shpfile_path)
    os.remove(shpfile_path[:-4]+".dbf")
    os.remove(shpfile_path[:-4] + ".prj")
    os.remove(shpfile_path[:-4] + ".shx")

#extract training samples from a clipped tif file and export them to .shp files
def extract_to_shpfile(input_clipped_tif,shpfile_path,img_size=264196,imgsi=512):
    # Name	Code
    # Cropland	10
    # Forest	20
    # Grassland	30
    # Shrubland	40
    # Wetland	50
    # Water	60
    # Tundra	70
    # Impervious surface	80
    # Bareland	90
    # Snow/Ice	100
    categorylist = [10,20,30,40,50,60,70,80,90,100]


    categorylist_name = ["Cropland","Forest","Grassland","Shrub",
                         "Wetland","Water_body","Tundra","Manmade",
                         "Bareland","Ice"]

    #["Bareland", "Cropland", "Forest", "Manmade", "Shrub", "Vegetation", "Water_body", "Wetland", "Ice"]

    # categorylist = [60]
    # categorylist_name = ["Water"]

    #randomly select some pixels
    #using rsgislib functions

    for i in range(len(categorylist)):

        if os.path.exists(input_clipped_tif):
            xcount, ycount, bands, data = multispectralim_to_array(input_clipped_tif, channel_last=False)
            if (int(xcount * ycount) <=(img_size)):
                gdalDS = gdal.Open(input_clipped_tif, gdal.GA_ReadOnly)
                band_1 = gdalDS.GetRasterBand(1)
                data_type = band_1.DataType

                # traverse data and find
                shpfile_directory = shpfile_path
                columns = gdalDS.RasterXSize
                rows = gdalDS.RasterYSize
                data_copy = data[0]
                # min_value = data_copy.min()

                category = categorylist[i]
                category = np.uint16(category)
                data_copy[data_copy != category] = 0
                data_copy[data_copy == category] = 1

                # export data_copy to an .tif file
                tifile_name = shpfile_directory + "/" + categorylist_name[i] + ".tif"
                if os.path.exists(tifile_name):
                    os.remove(tifile_name)
                write_tif_data(data_copy, tifile_name, gdalDS, rows, columns, data_type, band_count=1)

                outShp = shpfile_directory + "/" + categorylist_name[i] + ".shp"
                if os.path.exists(outShp):
                    os.remove(outShp)
                    os.remove(outShp[:-4] + ".shx")
                    os.remove(outShp[:-4] + ".prj")
                    os.remove(outShp[:-4] + ".dbf")

                # get shp
                rsgislib.vectorutils.polygoniseRaster(tifile_name, outShp, imgBandNo=1, maskImg=None, imgMaskBandNo=1)

                # delete .tif file
                os.remove(tifile_name)

                # delete the polygon whose pxlval is 0
                delete_noval_polygons(outShp)

                # rsgislib.vectorutils.exportPxls2Pts(tifile_name, outShp, 1, True)

                print(categorylist_name[i])
                # A utillity to polygonise a raster to a ESRI Shapefile.

                del gdalDS
                # os.remove("test.tif") #delete "test.tif"
            else:
                del xcount, ycount, bands, data
                os.remove(input_clipped_tif)


def extract_shpfiles(des_directory,threshold_tif_number=0):
    des_tif_files = os.listdir(des_directory)

    for desc_tif in des_tif_files:
        if "." in desc_tif:
            tif_number = float(desc_tif.split(".")[0])
            tif_number = int(tif_number)
            if True:#tif_number >= threshold_tif_number:
                input_clipped_tif = des_directory + "/" + str(tif_number) + "_Training" + "/" + str(tif_number) + "clipped.tif"
                shpfile_path = des_directory + "/" + str(tif_number) + "_Training"
                if os.path.exists(shpfile_path):
                    pass
                else:
                    os.makedirs(shpfile_path)
                extract_to_shpfile(input_clipped_tif, shpfile_path, img_size=270400)  # 270400=520*520


import shutil
shps_dire = "E:/Point_clouds_processing_yan/Areas_Pointclouds/Area1_Terencegdb/shps"
fath_dire = "E:/Point_clouds_processing_yan/Areas_Pointclouds/Area1_Terencegdb"
def generate_clipping_scripts(shps_dire,fath_dire):
    #generate a folder
    #shp1.shp, shp2.shp, shp2.shp, shp3.shp, shp4.shp
    shps = os.listdir(shps_dire)
    for shp in shps:
        if shp[-4:] == ".shp":
            folder_name = fath_dire+"/"+shp[:-4]+"_clippedimages"
            if os.path.exists(folder_name):
                shutil.rmtree(folder_name)
                # os.makedirs(folder_name)
            else:
                # os.makedirs(folder_name)
                pass

            if os.path.exists(fath_dire+"/"+"arcpro_scripts"):
                pass
            else:
                os.makedirs(fath_dire+"/"+"arcpro_scripts")

            txt_file = fath_dire+"/"+"arcpro_scripts/"+shp[:-4]+"_arcpro_clipping_script.txt"
            if os.path.exists(txt_file):
                os.remove(txt_file)
            else:
                pass
            with open(txt_file, 'w') as f:  # 设置文件对象
                str_import_arcpy = 'import arcpy'
                f.writelines(str_import_arcpy + '\n')  # 将字符串写入文件中
                tif_images = os.listdir(fath_dire)
                for tiffile in tif_images:
                    if tiffile[-4:] == ".tif":
                        if True:  # and desc_tif in list_tif:  # "int.tif"
                            # print(desc_tif)
                            shpfile = shps_dire + shp
                            destination_tif = fath_dire+"/"+tiffile
                            clipped_tif = fath_dire+"/"+shp[:-4]+"_clippedimages"+"/"+tiffile[:-4]+"_shp_"+shp[:-4]+".tif"
                            # get_mask(tif_path, shpfile)
                            shpfile = shps_dire+"/"+shp
                            desc_str = "desc = arcpy.Describe(" + "'" + shpfile + "'" + ")" + "\n"
                            f.writelines(desc_str)
                            f.writelines('ext = desc.extent' + '\n')
                            processing_str = "arcpy.Clip_management(" + "'" + destination_tif + "'" + "," + "str(ext)" + "," + "'" + clipped_tif + "'" + ", '#', 'NONE')"
                            # generate a batch processing script for clipping an destination image with .shp files
                            f.writelines(processing_str + '\n')

            print("the script has been geneated successfully!")

    print("TRUE")

if __name__ == "__main__":

    shps_dire = "E:\\Point_clouds_processing_yan\\Areas_Pointclouds\\Area1_Terencegdb\\Arrea1_C4647_final_OI_clip\\Tifimages\\Tifimages_90\\randomclipped_OI_clip_90"

    dest_tif_dire = "E:\\Point_clouds_processing_yan\\Areas_Pointclouds\\Area1_Terencegdb\\Arrea1_C4647_final_OIC_RGB_clip\\Tifimages\\Tifimages_90"

    generate_clipping_scripts(shps_dire, dest_tif_dire)