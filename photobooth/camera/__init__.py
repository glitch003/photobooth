#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Photobooth - a flexible photo booth software
# Copyright (C) 2018  Balthasar Reuter <photobooth at re - web dot eu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import logging

from PIL import Image, ImageOps
from io import BytesIO
import argparse
import os

import cv2
import numpy as np
import dlib

import time

from .PictureDimensions import PictureDimensions
from .. import StateMachine
from ..Threading import Workers

from ..face_detection import face_detection
from ..face_points_detection import face_points_detection
from ..face_swap import warp_image_2d, warp_image_3d, mask_from_points, apply_mask, correct_colours, transformation_from_points

import pickle

# Available camera modules as tuples of (config name, module name, class name)
modules = (
    ('python-gphoto2', 'CameraGphoto2', 'CameraGphoto2'),
    ('gphoto2-cffi', 'CameraGphoto2Cffi', 'CameraGphoto2Cffi'),
    ('gphoto2-commandline', 'CameraGphoto2CommandLine',
     'CameraGphoto2CommandLine'),
    ('opencv', 'CameraOpenCV', 'CameraOpenCV'),
    ('picamera', 'CameraPicamera', 'CameraPicamera'),
    ('dummy', 'CameraDummy', 'CameraDummy'))

dirpath = os.getcwd()
print("current directory is : " + dirpath)

# preload stuff
PREDICTOR_PATH = 'photobooth/models/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)


dst_img = cv2.imread('./destination.jpg')
dst_points_and_stuff = pickle.load( open( "dst_points_and_stuff.pkl", "rb" ) )

class Camera:

    def __init__(self, config, comm, CameraModule):

        super().__init__()

        self._comm = comm
        self._cfg = config
        self._cam = CameraModule

        self._cap = None
        self._pic_dims = None

        self._is_preview = self._cfg.getBool('Photobooth', 'show_preview')
        self._is_keep_pictures = self._cfg.getBool('Storage', 'keep_pictures')

        rot_vals = {0: None, 90: Image.ROTATE_90, 180: Image.ROTATE_180,
                    270: Image.ROTATE_270}
        self._rotation = rot_vals[self._cfg.getInt('Camera', 'rotation')]


    # takes in two stuff objects of type: {
    #     'points': <>,
    #     'shape': <>,
    #     'face': <>
    # }
    # and a destination image
    #
    # returns an image
    def face_swap_stuff(self, src_stuff, dst_stuff, dst_img):

        parser = argparse.ArgumentParser(description='FaceSwapApp')
        parser.add_argument('--warp_2d', default=False, action='store_true', help='2d or 3d warp')
        parser.add_argument('--correct_color', default=False, action='store_true', help='Correct color')
        parser.add_argument('--no_debug_window', default=True, action='store_true', help='Don\'t show debug window')
        parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle face order')
        args = parser.parse_args()

        src_points = src_stuff['points']
        src_shape = src_stuff['shape']
        src_face = src_stuff['face']

        dst_points = dst_stuff['points']
        dst_shape = dst_stuff['shape']
        dst_face = dst_stuff['face']

        h, w = dst_face.shape[:2]

        ### Warp Image
        if not args.warp_2d:
            ## 3d warp
            warped_src_face = warp_image_3d(src_face, src_points[:48], dst_points[:48], (h, w))
        else:
            ## 2d warp
            src_mask = mask_from_points(src_face.shape[:2], src_points)
            src_face = apply_mask(src_face, src_mask)
            # Correct Color for 2d warp
            if args.correct_color:
                warped_dst_img = warp_image_3d(dst_face, dst_points[:48], src_points[:48], src_face.shape[:2])
                src_face = correct_colours(warped_dst_img, src_face, src_points)
            # Warp
            warped_src_face = warp_image_2d(src_face, transformation_from_points(dst_points, src_points), (h, w, 3))

        ## Mask for blending
        mask = mask_from_points((h, w), dst_points)
        mask_src = np.mean(warped_src_face, axis=2) > 0
        mask = np.asarray(mask*mask_src, dtype=np.uint8)

        ## Correct color
        if not args.warp_2d and args.correct_color:
            warped_src_face = apply_mask(warped_src_face, mask)
            dst_face_masked = apply_mask(dst_face, mask)
            warped_src_face = correct_colours(dst_face_masked, warped_src_face, dst_points)

        ## Shrink the mask
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        ##Poisson Blending
        r = cv2.boundingRect(mask)
        center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))


        output = cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.NORMAL_CLONE)

        x, y, w, h = dst_shape
        dst_img_cp = dst_img.copy()
        dst_img_cp[y:y+h, x:x+w] = output
        output = dst_img_cp

        return output


    def get_uhh_points_of_faces(self, im, r=10):
        faces = face_detection(im)

        if len(faces) == 0:
            print('Detect 0 Face !!!')
            return []

        # uncomment this to make the face selector window show up
        # if len(faces) == 1:
        #     bbox = faces[0]
        # else:
        #     bbox = []

        #     def click_on_face(event, x, y, flags, params):
        #         if event != cv2.EVENT_LBUTTONDOWN:
        #             return

        #         for face in faces:
        #             if face.left() < x < face.right() and face.top() < y < face.bottom():
        #                 bbox.append(face)
        #                 break

        #     im_copy = im.copy()
        #     for face in faces:
        #         # draw the face bounding box
        #         cv2.rectangle(im_copy, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 1)
        #     cv2.imshow('Click the Face:', im_copy)
        #     cv2.setMouseCallback('Click the Face:', click_on_face)
        #     while len(bbox) == 0:
        #         cv2.waitKey(1)
        #     cv2.destroyAllWindows()
        #     bbox = bbox[0]

        bbox = faces[0]

        all_points = []

        for bbox in faces:

            points = np.asarray(face_points_detection(predictor, im, bbox))

            im_w, im_h = im.shape[:2]
            left, top = np.min(points, 0)
            right, bottom = np.max(points, 0)

            x, y = max(0, left-r), max(0, top-r)
            w, h = min(right+r, im_h)-x, min(bottom+r, im_w)-y

            all_points.append({
                'points': points - np.asarray([[x, y]]),
                'shape': (x, y, w, h),
                'face': im[y:y+h, x:x+w]
            })

        return all_points


    def startup(self):

        self._cap = self._cam()

        logging.info('Using camera {} preview functionality'.format(
            'with' if self._is_preview else 'without'))

        test_picture = self._cap.getPicture()
        if self._rotation is not None:
            test_picture = test_picture.transpose(self._rotation)

        self._pic_dims = PictureDimensions(self._cfg, test_picture.size)
        self._is_preview = self._is_preview and self._cap.hasPreview

        background = self._cfg.get('Picture', 'background')
        if len(background) > 0:
            logging.info('Using background "{}"'.format(background))
            bg_picture = Image.open(background)
            self._template = bg_picture.resize(self._pic_dims.outputSize)
        else:
            self._template = Image.new('RGB', self._pic_dims.outputSize,
                                       (255, 255, 255))

        self.setIdle()
        self._comm.send(Workers.MASTER, StateMachine.CameraEvent('ready'))

    def teardown(self, state):

        if self._cap is not None:
            self._cap.cleanup()

    def run(self):

        for state in self._comm.iter(Workers.CAMERA):
            self.handleState(state)

        return True

    def handleState(self, state):

        if isinstance(state, StateMachine.StartupState):
            self.startup()
        elif isinstance(state, StateMachine.GreeterState):
            self.prepareCapture()
        elif isinstance(state, StateMachine.CountdownState):
            self.capturePreview()
        elif isinstance(state, StateMachine.CaptureState):
            self.capturePicture(state)
        elif isinstance(state, StateMachine.AssembleState):
            self.assemblePicture()
        elif isinstance(state, StateMachine.TeardownState):
            self.teardown(state)

    def setActive(self):

        self._cap.setActive()

    def setIdle(self):

        if self._cap.hasIdle:
            self._cap.setIdle()

    def prepareCapture(self):

        self.setActive()
        self._pictures = []

    def capturePreview(self):

        if self._is_preview:
            while self._comm.empty(Workers.CAMERA):
                picture = self._cap.getPreview()
                if self._rotation is not None:
                    picture = picture.transpose(self._rotation)
                picture = picture.resize(self._pic_dims.previewSize)
                picture = ImageOps.mirror(picture)
                byte_data = BytesIO()
                picture.save(byte_data, format='jpeg')
                self._comm.send(Workers.GUI,
                                StateMachine.CameraEvent('preview', byte_data))

    def capturePicture(self, state):

        self.setIdle()
        picture = self._cap.getPicture()
        if self._rotation is not None:
            picture = picture.transpose(self._rotation)
        byte_data = BytesIO()
        picture.save(byte_data, format='jpeg')
        self._pictures.append(byte_data)
        self.setActive()

        if self._is_keep_pictures:
            self._comm.send(Workers.WORKER,
                            StateMachine.CameraEvent('capture', byte_data))

        if state.num_picture < self._pic_dims.totalNumPictures:
            self._comm.send(Workers.MASTER,
                            StateMachine.CameraEvent('countdown'))
        else:
            self._comm.send(Workers.MASTER,
                            StateMachine.CameraEvent('assemble'))

    def rotateCvImg(self, img):
        out = cv2.transpose(img)
        return cv2.flip(out,flipCode=0)
    # def rotateCvImg(self, img, deg):
    #     (h, w) = img.shape[:2]
    #     # calculate the center of the image
    #     center = (w / 2.0, h / 2.0)
    #     # rotate images for printer
    #     M = cv2.getRotationMatrix2D(center, deg, 1.0)
    #     return cv2.warpAffine(img, M, (h, w))

    def assemblePicture(self):

        start_time = time.time()

        self.setIdle()

        picture = self._template.copy()


        # dst_points_and_stuff = self.get_uhh_points_of_faces(dst_img)

        # pickle.dump( dst_points_and_stuff, open("dst_points_and_stuff.pkl", "wb") )

        print("Got points of dst in {} seconds".format(time.time() - start_time))

        image_pairs = []

        # swap faces
        for i in range(2):
            pil_image = Image.open(self._pictures[i])
            src_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            src_points_and_stuff = self.get_uhh_points_of_faces(src_img)
            print("Found {} src and {} dst images".format(len(src_points_and_stuff), len(dst_points_and_stuff)))
            print("Got points of src in {} seconds".format(time.time() - start_time))

            output = dst_img.copy()

            faces_to_clone = len(src_points_and_stuff)

            if len(dst_points_and_stuff) < len(src_points_and_stuff):
                faces_to_clone = len(dst_points_and_stuff)

            for i in range(faces_to_clone):
                print("swapping src face index {} into dst face index {}".format(i,i))
                output = self.face_swap_stuff(src_points_and_stuff[i], dst_points_and_stuff[i], output)

            image_pairs.append([src_img, output])

        watermark = cv2.imread('./watermark.jpg')
        # printer_row = np.hstack([
        #     self.rotateCvImg(image_pairs[0][0]),
        #     self.rotateCvImg(image_pairs[0][1]),
        #     self.rotateCvImg(image_pairs[1][0]),
        #     self.rotateCvImg(image_pairs[1][1]),
        #     self.rotateCvImg(watermark)
        # ])
        printer_row = self.rotateCvImg(np.vstack([
            image_pairs[0][0],
            image_pairs[0][1],
            image_pairs[1][0],
            image_pairs[1][1],
            watermark
        ]))
        printer_output = np.vstack([
            printer_row, printer_row
        ])
        print("Rotated printer image in {} seconds".format(time.time() - start_time))
        cv2_im = cv2.cvtColor(printer_output,cv2.COLOR_BGR2RGB)
        picture = Image.fromarray(cv2_im)#.rotate(270, expand=True)
        picture.save('/tmp/printme.jpg', format='jpeg')
        print("Saved printer file in {} seconds".format(time.time() - start_time))


        final_output = np.hstack([
            np.vstack([image_pairs[0][0], image_pairs[0][1]]),
            np.vstack([image_pairs[1][0], image_pairs[1][1]])
        ])

        cv2_im = cv2.cvtColor(final_output,cv2.COLOR_BGR2RGB)
        picture = Image.fromarray(cv2_im)

        print("Assembled in {} seconds".format(time.time() - start_time))


        # for i in range(4):
        #     shot = Image.open(self._pictures[i])
        #     resized = shot.resize(self._pic_dims.thumbnailSize)
        #     picture.paste(resized, self._pic_dims.thumbnailOffset[i % 2])

        # for i in range(self._pic_dims.totalNumPictures):
        #     shot = Image.open(self._pictures[i])
        #     resized = shot.resize(self._pic_dims.thumbnailSize)
        #     picture.paste(resized, self._pic_dims.thumbnailOffset[i])

        byte_data = BytesIO()
        picture.save(byte_data, format='jpeg')
        self._comm.send(Workers.MASTER,
                        StateMachine.CameraEvent('review', byte_data))
        self._pictures = []
