import os
import glob
import shutil

path = "../data"
champions = ["Chogath", "Ezreal", "Lucian", "Malzahar", "Morgana", "Poppy", "Reksai", "Senna", "Syndra", "Teemo"]

def create_dir(my_folder): # Create the folders for the training dataset and the test dataset
    directory = os.path.dirname(my_folder)
    if not os.path.exists(my_folder):
        os.makedirs(my_folder)
        print("making dir")



List_champions_pictures = []
for i in range(len(champions)) :
    List_champions_pictures.append(glob.glob(path+"/"+champions[i]+"/"+"*.bmp"))



for champ in champions : 
    create_dir(path+"/validation/validation_"+champ)
    create_dir(path+"/train/train_"+champ)



# It worked, I'm just too scared to execute it again

#if len(os.listdir(path+"/train_"+"teemo")) == 0 : # Just to make sure I use the program only once
#    for i in range(len(champions)) : # Take a fifth of the picture to the test folder and a fifth to the train folder
#        for num_pic in range(len(List_champions_pictures[i])) :
#            if num_pic % 5 == 0 :
#                shutil.move(List_champions_pictures[i][num_pic], path+"/test_"+champions[i] ) 
#            if num_pic % 5 == 1 :
#                shutil.move(List_champions_pictures[i][num_pic], path+"/train_"+champions[i] ) 
