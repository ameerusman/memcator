Extract the memcator.zip file and place the images of people you want to be recognized when scanning the images. Some sample images are already present. Preferably the images should be front facing. The images should contain only 1 person.

There are 3 utilities inside this zip file:
***memcator.insert***

This utility builds the database and takes file names as arguments and can be best searched using the following command:

find ./images/ -exec ./memcator/memcator/memcator.insert {} +;

***memcator.search***
This utility searchs inside the database for the relevant objects. The '+' is an AND operator and the '-' is a NOT operator for the prediactes. Always perform subsequent AND '+' operations first before performing the NOT '-' operations. Following are some example commands:

memcator/memcator.search ameer
memcator/memcator.search bottle
memcator/memcator.search ameer+bottle


***memcator.delete***
This utility removes all indexes done in the database.
memcator/memcator.delete

=====================================================
Please also refer to demo videos to see how it works.

=====================================================
Followinglibraries were used in Python 3.8
deepface==0.0.75
imageai==2.1.6
opencv_python==4.5.5.64
pandas==1.5.1
retina_face==0.0.12
retinaface==1.1.1






