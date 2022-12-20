import sys
import file_reader as fr
from settings import *

def find_valid_object_name(overall_objects):
    """

    Checks if the object/person to search is valid.

    """
    if(type(overall_objects) == str):
        object_to_search = overall_objects
        object_found = False
        for coco_object in classes_coco_people:
            if (coco_object == object_to_search):
                object_found = True
        if(object_found == False):
            print("Object/Person to find not in the list of possible detections....")
            print("The classes that can be detected are:")
            for coco_object in classes_coco_people:
                print(coco_object,end=", ")
            print("")
            return False

    else:
        for object_to_search in overall_objects:
            object_found = False
            for coco_object in classes_in_coco:
                if (coco_object == object_to_search):
                    object_found = True
            if(object_found == False):
                print("Object to find not in the list of possible detections")
                print("The objects that can be detecte are:")
                for coco_object in classes_in_coco:
                    print(coco_object,end=", ")
                print("")
                return False

    return True

def find_objects_in_list():
    """

    Find the objects given in the arguments inside the list.
    Also performing AND and NOT operations on predicates.

    """
    input_string = sys.argv[1]
    input_string = input_string.lower()
    overall_objects = []

    string_status = ""
    if(input_string.find('+') == -1 and input_string.find('-') == -1 ):
        overall_objects.append(input_string)

        string_status = "No signs"
        paths_to_print = [set()]
        list_index = -1
        json_file_data = fr.read_json(database_file_path)
        if (bool(json_file_data) and json_file_data != False):
            list_index = list_index + 1
            for json_data in json_file_data:
                if json_data["object"] == overall_objects[0]:
                    for data in json_data["file_path"]:
                        paths_to_print[list_index].add(data)

        if (len(paths_to_print) == 0) :
            return False,False,False
        
        if (find_valid_object_name(overall_objects) == False):
            return False,False,False

        return paths_to_print[0], overall_objects, False                

    elif(input_string.find('+') == -1):
    
        overall_objects = input_string.split('+')
        obj_not_to_find = str(overall_objects.pop()).split('-')

        obj_to_find = overall_objects.append(obj_not_to_find[0])
        # obj_not_to_find = [] #This created an issue by resetting the list which is used after this.
        obj_not_to_find = obj_not_to_find[1:]
        string_status = "Only minus"

        if (find_valid_object_name(overall_objects) == False):
            return False,False,False

        if (find_valid_object_name(obj_not_to_find) == False):
            return False,False,False

    elif (input_string.find('-') == -1):
        overall_objects = input_string.split('+')

        string_status = "Only plus"

        if (find_valid_object_name(overall_objects) == False):
            return False,False,False

    else:
        string_status = "Both signs"
            
        overall_objects = input_string.split('+')
        obj_not_to_find = str(overall_objects.pop()).split('-')

        overall_objects.append(obj_not_to_find[0])
        obj_not_to_find = obj_not_to_find[1:]

        if (find_valid_object_name(overall_objects) == False):
            return False,False,False

        if (find_valid_object_name(overall_objects) == False):
            return False,False,False

        if (find_valid_object_name(overall_objects) == False):
            return False,False,False

    paths_to_print = [set() for _ in range(len(overall_objects))]
    list_index = -1
    json_file_data = fr.read_json(database_file_path)
    if (bool(json_file_data) and json_file_data != False):
        for object in overall_objects:
            list_index = list_index + 1
            for json_data in json_file_data:
                if json_data["object"] == object:
                    for data in json_data["file_path"]:
                        paths_to_print[list_index].add(data)
                    break
        list_index
        path_set_initial = paths_to_print[0]

        first_entry = True
        for path_set in paths_to_print:
            if (first_entry == True):
                path_set = paths_to_print[0]
                first_entry = False
            path_set_initial = path_set.intersection(path_set_initial)

        if not (len(path_set_initial) == 0):
            if((string_status != "Only plus" and string_status != "No signs") or string_status == "Both signs"):
                for object in obj_not_to_find:
                    for json_data in json_file_data:
                        if json_data["object"] == object:
                            for data in json_data["file_path"]:
                                try:
                                    path_set_initial.remove(data)
                                    print("Removing {}".object)
                                except:
                                    pass        
                            break

        if (len(paths_to_print) == 0):
            return False,False,False
            
        else:
            if((string_status == "Only plus" or string_status == "No signs")):
                return path_set_initial, overall_objects, False
            else:
                return path_set_initial, overall_objects, obj_not_to_find
    else:
        print("Database is empty/not found.")
        return False,False,False

def main():        
    paths_to_print, objects_to_find, obj_not_to_find = find_objects_in_list()
    if(paths_to_print == False):

        exit()
    if (not paths_to_print == False):
        if(obj_not_to_find == False):
            print("Below are the files containing image(s) of: " + str(objects_to_find) )
            for path_to_print in paths_to_print:
                print(str(path_to_print))
            return

        if (len(obj_not_to_find) == 0 ):
            print("Below are the files containing image(s) of: " + str(objects_to_find) )
        else:
            print("Below are the files containing image(s) of: " + str(objects_to_find) + ", but do not contain image(s) of: " + str(obj_not_to_find))
        for path_to_print in paths_to_print:
            print(str(path_to_print))

if __name__ == "__main__":
    main()
