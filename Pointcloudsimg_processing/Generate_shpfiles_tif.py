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


def batch_processing(des_directory):
    des_tif_files = os.listdir(des_directory)
    for desc_tif in des_tif_files:
        suffix = desc_tif[-4:]  # desc_tif[-7:]
        if suffix == ".tif":  # "int.tif"
            # print(desc_tif)
            # get_mask(des_directory+ "/" + desc_tif, des_directory + "/" + desc_tif[:-4]+"_Training"+"/"+desc_tif[:-4]+".shp")
            get_mask(des_directory + "/" + desc_tif,
                     des_directory + "/" + desc_tif[:-4] + ".shp")
            # get_mask(img_path, shp_path)

    print("finished")

if __name__ == "__main__":
    # img_path = "H:/InformationTransmission_entropy/Images/Sentinel2/Sentinel2_Jiangsu/Tifimages/20191115T023951_20191115T024840_T51STR_Jiangsu-0000016384-0000016384.tif"
    # shp_path = "H:/InformationTransmission_entropy/Images/Sentinel2/Sentinel2_Jiangsu/Tifimages/mask_use.shp"
    #
    # img_path = "G:/GEE_Images/Sentinel2_Guangdong/Tifimages/20201230T030129_20201230T030613_T49QHG_Guangdong-0000000000-0000016384.tif"
    # shp_path = "G:/GEE_Images/Sentinel2_Guangdong/Tifimages/mask_use.shp"
    #des_directory = "J:\\Sentinel2_Nanjing\\Tifimages_clipped\\20201023T024751_20201023T024754_T50SPA_Nanjing-0000000000-0000000000" #the directory for .tif files
    #des_directory = "E:\\Point_clouds_processing_yan\\Areas_Pointclouds\\Area1_Terencegdb\\Arrea1_C4647_final_OI_clip\\Tifimages\\Tifimages_0\\randomclipped_OI_clip_0"

    des_directory = "E:\\Point_clouds_processing_yan\\Areas_Pointclouds\\Area3_Terencegdb\\Area3_C6063_OI_RGB_clip\\Tifimages\\Tifimages_0\\randomclipped_OI_clip_0"
    batch_processing(des_directory)
    # get_mask(img_path,shp_path)
