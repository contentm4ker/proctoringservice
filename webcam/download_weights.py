import wget

if __name__ == '__main__':    
    wget.download('https://pjreddie.com/media/files/yolov3.weights', out='models/yolov3.weights')
