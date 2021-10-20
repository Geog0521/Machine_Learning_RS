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
    path1 = "E:\\Point_clouds_processing_yan\\Ecognition\\Area1_randomclipped_OI_clip_0"
    path2 = "E:\\Point_clouds_processing_yan\\Ecognition\\Area1_randomclipped_OI_clip_0"
    print("TRUE")
    suffix = ""
    landcover_types = ["Bare_soil_shelterwood_stand", "Open_mixedwood", "Natural_forest_stand", "Grassland",
                       "Clearcut_stand", "Road", "Wetland", "Water_body"]

    generate_shpfiles_training(path1, path1, suffix, landcover_types)
    generate_shpfiles_training(path2, path2, suffix, landcover_types)