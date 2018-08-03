# import the necessary packages
import matplotlib
matplotlib.use('Agg')
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import os 
import subprocess
import os.path

# TODO: Currently the bounding boxes are drawn without showing the object's class on top of the bounding box
def runDetection(path,orgName):
    #note: orgName is "original name"
    print "I MADE IT HERE"
    # Make sure that caffe is on the python path:
    caffe_root = '/home/paperspace/dev/caffe'  # this file is expected to be in {caffe_root}/examples
    import os
    os.chdir(caffe_root)
    import sys
    sys.path.insert(0, 'python')

    import caffe
    caffe.set_device(0)
    caffe.set_mode_gpu()

    from google.protobuf import text_format
    from caffe.proto import caffe_pb2

    # load PASCAL VOC labels
    labelmap_file = '/home/paperspace/dev/caffe/data/VOC0712/labelmap_voc.prototxt'
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)

    def get_labelname(labelmap, labels):
        num_labels = len(labelmap.item)
        labelnames = []
        if type(labels) is not list:
            labels = [labels]
        for label in labels:
            found = False
            for i in xrange(0, num_labels):
                if label == labelmap.item[i].label:
                    found = True
                    labelnames.append(labelmap.item[i].display_name)
                    break
            assert found == True
        return labelnames
    #load the input video
    print("[INFO] starting video file thread...")
    fvs = FileVideoStream(path).start()
    time.sleep(1.0)
    
    model_def = '/home/paperspace/dev/caffe/models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
    model_weights = '/home/paperspace/dev/caffe/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'

    net = caffe.Net(model_def,      # defines the structure of the model
                        model_weights,  # contains the trained weights
                        caffe.TEST)     # use test mode (e.g., don't perform dropout)
    
    if os.path.isfile("/home/paperspace/dev/caffe/uploadResults/output.wmv"):
        os.remove("/home/paperspace/dev/caffe/uploadResults/output.wmv")
        
    #the output file is stored as wmv initially since the video writer can't handle mp4 formats, I change the wmv format to mp4 
    #later on
    uploadResultPath = "/home/paperspace/dev/caffe/uploadResults/output.wmv"
    # start the FPS timer
    fps = FPS().start()
    frame = fvs.read()
    height, width, layers = frame.shape
    #we use 20 as our fps, adjust acoording to SSD!, 450 = width, 253= height, create the output file to  write to (450, 253)
    video = cv2.VideoWriter(uploadResultPath, cv2.cv.CV_FOURCC('M','P','4','2'),30, (width,height))
    
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104,117,123])) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    counter = 0
    # loop over frames from the video file stream
    while fvs.more():
        frame = fvs.read()
        print "NEW FRAME LOADED"
        inputPath = "/home/paperspace/dev/caffe/uploadResults/frame"+str(counter)+".jpg"
        cv2.imwrite(inputPath, frame) #path instead of inputPath
        
        
        # set net to batch size of 1
        image_resize = 300
        net.blobs['data'].reshape(1,3,image_resize,image_resize)

        image = caffe.io.load_image(inputPath)


        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.4]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]


        #plt.save()


        #image is an array and plt.imshow will turn it into an image datatype
        #add_image returns the image and adds it to currentAxis (axes DT)! it takes data type image as an attribute
        goodFrame = 0
        
        for i in xrange(top_conf.shape[0]):
            goodFrame = 1
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]*100
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            display_txt = '%s: %.2f'%(label_name, score)+"%"
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            cv2.rectangle(frame,(xmin,ymax),(xmax,ymin),(0,0,255),2)
            
            
           # cv2.putText(image,'{:s} {:.3f}'.format(label_name, score),(int(bbox[0]),int(bbox[1]-2)),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255))#,cv2.CV_AA)

           # currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
           # currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})


        if goodFrame == 1:
            print "DETECTION(S) OCCURED ON THE CURRENT FRAME"
            outPath = "/home/paperspace/dev/caffe/uploadResults/outFrame"+str(counter)+".jpg"
            cv2.imwrite(outPath, frame)
            resizedImage = cv2.resize(cv2.imread(outPath), (width, height), interpolation = cv2.INTER_AREA)
            cv2.imwrite(outPath, resizedImage)
            video.write(cv2.imread(outPath))
            os.remove(outPath)
            os.remove(inputPath)
            counter = counter +1

        else:
            print "NO DETECTION OCCURED ON THE CURRENT FRAME"
            video.write(cv2.imread(inputPath))
            os.remove(inputPath)
            counter = counter +1
 
        
        
        fps.update()

    # stop the timer and release the video
    video.release()
    fps.stop()
    if os.path.isfile('/home/paperspace/dev/caffe/uploadResults/'+orgName+'.mp4'):
        os.remove('/home/paperspace/dev/caffe/uploadResults/'+orgName+'.mp4')
        
        #here I change the output format from the wmv file to an mp4 format with the file's original name:
        os.system('cd /home/paperspace/dev/caffe/uploadResults/; ffmpeg -i output.wmv -s 1920x1080 '+orgName+'.mp4')
        uploadResultPath = '/home/paperspace/dev/caffe/uploadResults/'+orgName+'.mp4'
        
    else:
        #here I change the output format from the wmv file to an mp4 format with the file's original name:
        os.system('cd /home/paperspace/dev/caffe/uploadResults/; ffmpeg -i output.wmv -s 1920x1080 '+orgName+'.mp4')
        uploadResultPath = '/home/paperspace/dev/caffe/uploadResults/'+orgName+'.mp4'

    print("[INFOcd] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    fvs.stop()
    os.remove('/home/paperspace/dev/caffe/uploads/'+orgName+'.mp4')
    return uploadResultPath

