#Memcator


import os
import time
import cv2
import sys
import logging
import warnings
import logging
from deepface import DeepFace
from retinaface import RetinaFace
import json

import flow_control
from settings import *


def check_file_extension(file_name):
    """

    Checks if the extension of the file is valid

    """
    file_type_valid = False
    try:
        if (file_name):
            file_name_to_add = os.path.basename(file_name)
            split_tup = os.path.splitext(file_name_to_add)
            for image_file_type in image_file_types:
                
                if (split_tup[1] == image_file_type):
                    file_type_valid = True
                    return True
                    break
            if (file_type_valid == False):
                return False  

    except:
        pass

    assert os.path.exists(file_name)

    return False

# =============


def get_detected_face (face):
    """
    Takes image file path and returns the cropped face in the image.

    """
    fd = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_img = cv2.imread(face, 0)
    fr = fd.detectMultiScale(face_img)
    for (x,y,width,height) in fr:
        cv2.rectangle(face_img, (x,y), (x + width, y+height), (255,255,77), 10)  
    croppedFace = face_img[x:x+width,y:y+height]
    return croppedFace


def memcator_init(path , model_name):
    """

    Initializes the model passed in the argument using the imageai library.

    """
    from imageai.Detection import ObjectDetection

    flow_control.log("Memcator")
    t1 = time.time()
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(path , model_name ))
    detector.loadModel()
    t2 = time.time()
    string = "Time to load model: " + str(t2-t1) + " seconds"
    flow_control.log(string)
    return detector
    
def memcator_general_detect(path, general_model , input_image , output_image):
    """
    
    Initializes the model passed in the argument using the imageai library.

    """
    t1 = time.time()
    detections = general_model.detectObjectsFromImage(input_image=os.path.join(path , input_image),output_image_path=os.path.join(path , output_image))
    # detections = general_model.detectObjectsFromImage(input_image=input_image1, output_image_path=os.path.join(path , output_image))
    # detections = general_model.detectObjectsFromImage(input_image=os.path.join(path , input_image), output_image_path=os.path.join(path , output_image))
    # detections = general_model.detectObjectsFromImage(input_image=input_image)
    t2 = time.time()
    string = "Time to perform general detections: " + str(t2-t1) + " seconds"
    flow_control.log(string)
    return detections


def convert_to_list(string):
    """
    
    To change a string into a list of strings (this was to avoid type inconsistency).

    """
    li = list(string.split())
    return li

def prepare_json_data_single(object,file_path,json_data):
    """
    
    JSON data is prepared, appended and returned.

    """
    json_obj = {}
    
    json_obj["object"] = object # This means to be checked
    json_obj["file_path"] = []
    json_obj["file_path"].append(file_path)

    json_data.append(json_obj)
    return json_data


def update_json_data(objects,json_file_data,img_file_path):
    """
    
    The JSON data is updated. It searches for already present paths in the JSON to avoid duplicates.

    """
    file_paths = []
    file_paths_to_add = img_file_path
    
    for object in objects:
        found_object = False
        json_data_index = 0

        if bool(json_file_data):
            for json_data in json_file_data:
                if json_data["object"] == object:
                    found_object = True
                    file_paths = json_file_data[json_data_index]["file_path"]

                    if type(file_paths) is str:
                        file_paths = convert_to_list(file_paths)
                    file_paths.append(file_paths_to_add)
                    json_file_data[json_data_index]["file_path"] = file_paths
                    break
                json_data_index = json_data_index + 1

            if (found_object == False):
                    json_file_data = prepare_json_data_single(object,img_file_path,json_file_data)    
        else:
            json_file_data = []
            json_file_data = prepare_json_data_single(object,img_file_path,json_file_data)
    return json_file_data

def read_json(file_path):
    """
    
    Reads the JSON file if found. If not found, it returns an empty dictionary.

    """
    try:
        with open(file_path) as f:
            data = json.load(f)
        return data

    except:
        json_obj = {}
    return json_obj

def read_json_with_check(file_path):
    """
    
    Reads the JSON file but does not return a False if file not found.

    """
    try:
        with open(file_path) as f:
            data = json.load(f)
        return data

    except:
        return False


def create_or_update_json_file_with_data(path,obj_list):
    """
    
    Updates the JSON file. If file is not present, it creates and popilates it.

    """
    try:
        with open(database_file_path,'w') as json_file:
            json.dump(obj_list,json_file,indent=4)
    except:
        print("An error occured while writting the Database.")
        
def update_database_with_objects(img_file_path, object_list):
    """
    
    Updates the objects inside the data.

    """   

    if (DATABASE_FORMAT == DatabaseFormat.JSON):
        json_file_data = read_json(database_file_path)
        json_updated_data = update_json_data(object_list,json_file_data,img_file_path)
        create_or_update_json_file_with_data(path=database_file_path, obj_list=json_updated_data)

def memcator_process_detections(detection_results,image_path):
    """
    
    Processes the detections inside memcator

    """
    objects_found = set()
    for eachObject in detection_results:
        if(int(eachObject["percentage_probability"]) > DETECTION_PROBABILITY):
            object_without_space = eachObject["name"].replace(" ","")
            objects_found.add(object_without_space )

    return objects_found

def memcator_facial_detect(image_path, objects_set):
    """
    
    Performs face extractions, aligns them and performs comaprison with images of people
    in known_people_images folder.

    """
    t1 = time.time()
    faces = RetinaFace.extract_faces(img_path = image_path, align = True)
    t2 = time.time()
    string = "Time to extract faces: " + str(t2-t1) + " seconds"
    flow_control.log(string)

    face_counter = 0
    for face in faces:
        # print("Faces Found" + str(face_counter))
        face_counter = face_counter + 1
        # cv2.imwrite("face_"+str(face_counter)+".jpg",face)
        for root, dirs, files in os.walk(knwn_ppl_imgs_path):
            for file in files:
                if file.endswith(image_file_types):
                    t1 = time.time()
                    detection_result = DeepFace.verify(img1_path = os.path.join(root, file), 
                    img2_path = face, model_name = model,  
                    distance_metric = metric, enforce_detection=False)
                    t2 = time.time()
                    string = "Time to recognize a face: " + str(t2-t1) + " seconds"
                    flow_control.log(string)

                if (detection_result["verified"] == True):
                    # split_string = file.split('.')
                    split_string = os.path.splitext(file)
                    person_name = split_string[0]
                    objects_set.add(person_name)
                
                elif (detection_result["verified"] == False ):
                    pass
                    

    return objects_set

def find_duplicate_in_db(database_file,image_path_to_check):
    """
    
    Searches if the file path is already present in the database.

    """
    json_file_data = read_json_with_check(database_file)
    if(json_file_data != False):
        if(bool(json_file_data)):
            for json_data in json_file_data:
                for json_file_paths in json_data["file_path"]:

                    if (json_file_paths == image_path_to_check):
                        return True
    
    else:
        return "Not inside the Database."
    return False
    
    


# =========== Main Function Starts Here ============
def main():

    execution_path = model_path
    # input_image_with_path = os.path.abspath(input_image)

    general_model = memcator_init(execution_path , model_name)
    output_image =  os.path.join(model_path , "tmp.jpg" )
    
    files_count = len(sys.argv) - 1
    print("Total files found: ",end="")
    print(files_count)
    file_counter = 1
    while(file_counter <= files_count):
        print(sys.argv[file_counter])
        input_image = os.path.abspath(sys.argv[file_counter])
        file_counter = file_counter + 1
        if (check_file_extension(input_image) == False):
            print("Not a valid extension.")
            continue
        
        if (find_duplicate_in_db(database_file_path,input_image) != True):
            print("Analyzing file: "+ input_image)
            detections = memcator_general_detect(execution_path, general_model , input_image , output_image)
            
            objects_set = memcator_process_detections(detections,input_image)

            if(person_string in objects_set): # If a person is found inside the image
                objects_set = memcator_facial_detect(input_image, objects_set)

            # Only proceed forward if there are any objects found
            if(bool(objects_set)):
                update_database_with_objects(input_image, objects_set)

            else:
                # print("Found no objects.")
                pass
        else:
            print("File already in scanned.")
            pass

# ============= Calling the main function ==========
if __name__ == "__main__":
    warnings.filterwarnings("ignore") # This is to remove warnings by tensorflow
    logging.getLogger("tensorflow").setLevel(logging.ERROR) # This is to remove warnings by tensorflow
    main()