import dlib

## Face detection
def face_detection(detector, img,upsample_times=1):
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.

    faces = detector(img, upsample_times)

    return faces
