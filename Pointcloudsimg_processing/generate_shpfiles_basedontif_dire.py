import osgeo.ogr as ogr
import osgeo.osr as osr
import os

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")

def generate_shpfiles_training(path,overall_path,suffix,landcover_types):
    files = os.listdir(path)  # get all available shapefiles
    for tifFile in files:
        tifFile_path = os.path.join(overall_path, tifFile)
        # xmlpath = xmlpath+ district_list[k]
        split_str = tifFile_path.split('_')
        tif_suffix = suffix+".tif"
        print(split_str[len(split_str)-1])
        name_str = tifFile_path[-4:]
        # if split_str[len(split_str)-1]==tif_suffix:
        if name_str == tif_suffix:
            print("generating a folder successfully!")
            training_shpfiles = tifFile_path[:-4] + "_Training"
            mkdir(training_shpfiles)
            print(training_shpfiles)
            generate_shpfiles(training_shpfiles,landcover_types)

        # if os.path.splitext(tifFile_path)[1] == ".tif":

    # print("repair all images successfully!")

def generate_shpfiles(training_shpfiles_path,landcover_types):

    for i in range(len(landcover_types)):
        driver = ogr.GetDriverByName("ESRI Shapefile")
        data_source = driver.CreateDataSource(training_shpfiles_path + "\\"+landcover_types[i] + ".shp")
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = data_source.CreateLayer("test", srs, ogr.wkbPolygon)

        field_name = ogr.FieldDefn("class",ogr.OFTInteger)
        layer.CreateField(field_name)
        feature = ogr.Feature(layer.GetLayerDefn())
        # feature.SetField("class",i)
        driver = None
        layer = None

    print("all .shp files have been generated,please check out your folders")

    return True

if __name__ == "__main__":

    suffix = ""
    landcover_types=["Bare_soil_shelterwood_stand", "Open_mixedwood", "Natural_forest_stand", "Grassland", "Clearcut_stand", "Road","Wetland","Water_body"]
    # landcover_types= ["Bareland","Cropland","Forest","Manmade","Shrub","Vegetation","Water_body","Wetland","Ice"]

#     landcover_types = ["L1_Com_high_rise","L2_Com_midrise","L3_Com_low_rise","L4_Open_high_rise","L5_Open_mid_rise","L6_Open_low_rise","L7_Light_low_rise",
# "L8_Large_low_rise","L9_Sparsely_built","LA_Dense_trees","LB_Scatt_trees","LC_Bush_scrub","LD_low_plants","LE_barerock_paved","LF_Bare_soil_sand",
# "LG_Water"]
#     landcover_types = ["Bareland", "Cropland", "Forest", "Manmade", "Shrub", "Vegetation", "Water_body", "Wetland", "Ice"]
    # landcover_types = ["Shrub"]
    # landcover_types = ["Bareland", "Cropland", "Forest", "Manmade", "Shrub", "Vegetation", "Water_body", "Wetland", "Ice"]

    father_path_area3 = "E:\Point_clouds_processing_yan\\Areas_Pointclouds\\Area3_Terencegdb\\Area3_C6063_OIC_RGB_clip\\Tifimages\\Tifimages"
    father_path_area2 = "E:\Point_clouds_processing_yan\\Areas_Pointclouds\\Area2_Terencegdb\\Area2_C6263_OIC_RGB_clip\\Tifimages\\Tifimages"
    father_path_area1 = "E:\Point_clouds_processing_yan\\Areas_Pointclouds\\Area1_Terencegdb\\Arrea1_C4647_final_OIC_RGB_clip\\Tifimages\\Tifimages"
    scale_pathlist = ["0","90","180","270"]
    # Area_OI = "Area2_C6263_OI_RGB_clip"
    # Area_OIC = "Area2_C6263_OIC_RGB_clip"

    for scale in scale_pathlist:
        # path = father_path+"\\"+Area_OI+scale+"\\"+"Tifimages_clipped\\"+Area_OI+scale
        path = father_path_area1+"_"+ scale+"\\"+"randomclipped_OIC_clip"+"_" + scale
        generate_shpfiles_training(path, path, suffix, landcover_types)
    # print("generating .shp files successfully!")



    # father_path = "E:\\Point_clouds_processing_yan\\Areas_Pointclouds\\Area3_Terencegdb"
    # scale_pathlist = ["","_1m","_2m","_5m","_10m","_20m"]
    # Area_OI = "Area3_C6063_OI_RGB_clip"
    # Area_OIC = "Area3_C6063_OIC_RGB_clip"
    # for scale in scale_pathlist:
    #     path = father_path+"\\"+Area_OI+scale+"\\"+"Tifimages_clipped\\"+Area_OI+scale
    #     generate_shpfiles_training(path, path, suffix, landcover_types)
    #
    #     path = father_path + "\\" + Area_OIC + scale + "\\" + "Tifimages_clipped\\" + Area_OIC + scale
    #     generate_shpfiles_training(path, path, suffix, landcover_types)

    print("generating .shp files successfully!")


