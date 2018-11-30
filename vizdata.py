import cv2
import pickle

data = pickle.load(open("/Users/balazs/real_data/data_training.pkl", 'rb'))

for image, truth in data:
    if len(truth) > 50:
        print(truth)
        cv2.imshow("image", image)
        cv2.waitKey(0)