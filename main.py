import blobconverter
import cv2
import depthai as dai
import numpy as np
import os

from time import time, ctime

nameNN = "vehicle-detection-adas-0002"

def createPipeline():
    print("Creating Pipeline...")
    pipeline = dai.Pipeline()

    print("Creating Color Camera...")
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(672, 384)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setFps(30)
    cam.setVideoSize(1080, 1080)
    cam.setInterleaved(False)

    cam_xout = pipeline.createXLinkOut()
    cam_xout.setStreamName("frame")
    cam.video.link(cam_xout.input)

    print("Creating Detection Neural Network...")
    vehicleDetNN = pipeline.createMobileNetDetectionNetwork()
    vehicleDetNN.setConfidenceThreshold(0.75)
    vehicleDetNN.setBlobPath(blobconverter.from_zoo(
        name = nameNN,
        shaves = 6
    ))

    cam.preview.link(vehicleDetNN.input)

    print("Creating Object Tracker...")
    objectTracker = pipeline.createObjectTracker()
    #objectTracker.setDetectionLabelsToTrack([1])
    objectTracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
    objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

    print("Linking...")
    vehicleDetNN.passthrough.link(objectTracker.inputDetectionFrame)
    vehicleDetNN.passthrough.link(objectTracker.inputTrackerFrame)
    vehicleDetNN.out.link(objectTracker.inputDetections)

    passXOut = pipeline.createXLinkOut()
    passXOut.setStreamName("passOut")
    objectTracker.passthroughTrackerFrame.link(passXOut.input)

    trackletsXOut = pipeline.createXLinkOut()
    trackletsXOut.setStreamName("tracklets")
    objectTracker.out.link(trackletsXOut.input)

    print("Pipeline Successfully Created.")
    return pipeline

t = time()
timeOfCreation = ctime(t)
timeOfCreation = str(timeOfCreation)


with dai.Device(createPipeline()) as device:
    frameQ = device.getOutputQueue(name = "frame", maxSize = 1, blocking = False)
    trackletsQ = device.getOutputQueue(name = "tracklets", maxSize = 1, blocking = False)
    passQ = device.getOutputQueue(name = "passOut", maxSize = 1, blocking = False)

    vehicles = []
    totalVehicles = 0
    vehicleDict = []

    vehicleDir = ""

    leftBox = [0, 540]
    rightBox = [540, 1080]
    sensorBox = [460, 620] #Difference of 160

    while True:

        nnIn = trackletsQ.tryGet()
        if nnIn is not None:
            seq = passQ.get().getSequenceNum()



            frame = frameQ.get().getCvFrame()

            for t in nnIn.tracklets:
                #roi is Region Of Interest
                roi = t.roi.denormalize(frame.shape[1], frame.shape[0]) #width, height

                bbox = [int(roi.topLeft().x), int(roi.topLeft().y), int(roi.bottomRight().x), int(roi.bottomRight().y)]

                vehicle = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                fh, fw, fc = vehicle.shape
                frameh, framew, framec = frame.shape

                midpoint = (bbox[0]+bbox[2])/2
                midpoint = int(round(midpoint))
                
                vehicleID = str(t.id)
                vehicleAge = int(t.age)
                vehicleStatus = str(t.status)
                vehicleLabel = str(t.label)
                #print(vehicleID)
                #TrackingStatus.NEW, TrackingStatus.TRACKED, TrackingStatus.LOST, TrackingStatus.REMOVED

                if vehicleStatus == "TrackingStatus.NEW" and bbox[2] <= leftBox[1]:
                    vehicleDir = "left"
                elif vehicleStatus == "TrackingStatus.NEW" and bbox[0] >= rightBox[0]:
                    vehicleDir = "right"

                if vehicleID not in vehicles and sensorBox[0] <= midpoint <= sensorBox[1]:
                    t = time()
                    currentTime = ctime(t)
                    vehicles.append(vehicleID)
                    totalVehicles += 1
                    vehicleDict.append(("Vehicle #{}".format(totalVehicles), str(currentTime), vehicleDir))


                if vehicleID in vehicles and vehicleStatus == "TrackingStatus.REMOVED" or 360 > midpoint < 720:
                    if vehicleID in vehicles and len(vehicles) != 0:
                        vehicles.remove(vehicleID)
                    
                width = int(roi.bottomRight().x) - int(roi.topLeft().x)

                #create bounding box square
                cv2.putText(frame, str(bbox[0]), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 3)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (100, 50, 189), 3)
                cv2.line(frame, (midpoint, bbox[1]), (midpoint, bbox[3]), (0, 0, 255), 3)
            print(vehicles)
            print(totalVehicles)
            cv2.rectangle(frame, (sensorBox[0], 0), (sensorBox[1], 1080), (255, 0, 0), 3)
            cv2.rectangle(frame, (leftBox[0], 0), (leftBox[1], 1080), (255, 0, 255), 3)
            cv2.rectangle(frame, (rightBox[0], 0), (rightBox[1], 1080), (0, 255, 255), 3)
            cv2.imshow("Frame", cv2.resize(frame, (900, 900)))


        if cv2.waitKey(1) == ord("q"):
            data = open("DataDir/Data collected @ {}.txt".format(timeOfCreation), "x")
            data = open("DataDir/Data collected @ {}.txt".format(timeOfCreation), "w")
            for item in vehicleDict:
                data.write("{}-{}-{}".format(item[0], item[1], item[2]) + "\n")
            data.close()
            cv2.destroyAllWindows
            break