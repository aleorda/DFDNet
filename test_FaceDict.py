import logging
import os
import sys

import cv2
import dlib
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from skimage import transform as trans

from data.image_folder import make_dataset
from models import create_model
from options.test_options import TestOptions
from util.visualizer import save_crop

sys.path.append('FaceLandmarkDetection')
import face_alignment

logger = logging.getLogger(__name__)


###########################################################################
################# functions of crop and align face images #################
###########################################################################
def get_5_points(img):
    dets = detector(img, 1)

    if len(dets) == 0:
        return None

    areas = []
    if len(dets) > 1:
        logger.warning('\tMore than one face is detected. In this version, we only handle the largest one.')

    for j in range(len(dets)):
        area = (dets[j].rect.right() - dets[j].rect.left()) * (dets[j].rect.bottom() - dets[j].rect.top())
        areas.append(area)
    ins = areas.index(max(areas))
    shape = sp(img, dets[ins].rect)
    single_points = []
    for j in range(5):
        single_points.append([shape.part(j).x, shape.part(j).y])
    return np.array(single_points)


def align_and_save(img_path, save_path, save_input_path, save_param_path, upsample_scale=2):
    out_size = (512, 512)
    img = dlib.load_rgb_image(img_path)
    h, w, _ = img.shape
    source = get_5_points(img)

    if source is None:
        logger.error('\tNo face is detected')
        return

    tform = trans.SimilarityTransform()
    tform.estimate(source, reference)
    m = tform.params[0:2, :]
    crop_img = cv2.warpAffine(img, m, out_size)
    io.imsave(save_path, crop_img)  # save the crop and align face
    io.imsave(save_input_path, img)  # save the whole input image
    tform2 = trans.SimilarityTransform()
    tform2.estimate(reference, source * upsample_scale)
    # inv_m = cv2.invertAffineTransform(m)
    np.savetxt(save_param_path, tform2.params[0:2, :], fmt='%.3f')  # save the inverse affine parameters


def reverse_align(input_path, face_path, param_path, save_path, upsample_scale=2):
    # out_size = (512, 512)
    input_img = dlib.load_rgb_image(input_path)
    h, w, _ = input_img.shape
    face512 = dlib.load_rgb_image(face_path)
    inv_m = np.loadtxt(param_path)
    inv_crop_img = cv2.warpAffine(face512, inv_m, (w * upsample_scale, h * upsample_scale))
    mask = np.ones((512, 512, 3), dtype=np.float32)  # * 255
    inv_mask = cv2.warpAffine(mask, inv_m, (w * upsample_scale, h * upsample_scale))
    upsample_img = cv2.resize(input_img, (w * upsample_scale, h * upsample_scale))
    inv_mask_erosion_remove_border = cv2.erode(inv_mask, np.ones((2 * upsample_scale, 2 * upsample_scale),
                                                                 np.uint8))  # to remove the black border
    inv_crop_img_remove_border = inv_mask_erosion_remove_border * inv_crop_img
    total_face_area = np.sum(inv_mask_erosion_remove_border) // 3
    w_edge = int(total_face_area ** 0.5) // 20  # compute the fusion edge based on the area of face
    erosion_radius = w_edge * 2
    inv_mask_center = cv2.erode(inv_mask_erosion_remove_border, np.ones((erosion_radius, erosion_radius), np.uint8))
    blur_size = w_edge * 2
    inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
    merge_img = inv_soft_mask * inv_crop_img_remove_border + (1 - inv_soft_mask) * upsample_img
    io.imsave(save_path, merge_img.astype(np.uint8))


###########################################################################
################ functions of preparing the test images ###################
###########################################################################
def add_up_sample(img):
    return img.resize((512, 512), Image.BICUBIC)


def get_part_location(part_path, name):
    landmarks = []
    if not os.path.exists(os.path.join(part_path, name + '.txt')):
        print(os.path.join(part_path, name + '.txt'))
        print('\t################ No landmark file')
        return 0
    with open(os.path.join(part_path, name + '.txt'), 'r') as f:
        for line in f:
            tmp = [np.float(i) for j in line.split(' ') if i != '\n']
            landmarks.append(tmp)
    landmarks = np.array(landmarks)
    map_le = list(np.hstack((range(17, 22), range(36, 42))))
    map_re = list(np.hstack((range(22, 27), range(42, 48))))
    map_no = list(range(29, 36))
    map_mo = list(range(48, 68))
    try:
        # left eye
        mean_le = np.mean(landmarks[map_le], 0)
        l_le = np.max((np.max(np.max(landmarks[map_le], 0) - np.min(landmarks[map_le], 0)) / 2, 16))
        location_le = np.hstack((mean_le - l_le + 1, mean_le + l_le)).astype(int)
        # right eye
        mean_re = np.mean(landmarks[map_re], 0)
        l_re = np.max((np.max(np.max(landmarks[map_re], 0) - np.min(landmarks[map_re], 0)) / 2, 16))
        location_re = np.hstack((mean_re - l_re + 1, mean_re + l_re)).astype(int)
        # nose
        mean_no = np.mean(landmarks[map_no], 0)
        l_no = np.max((np.max(np.max(landmarks[map_no], 0) - np.min(landmarks[map_no], 0)) / 2, 16))
        location_no = np.hstack((mean_no - l_no + 1, mean_no + l_no)).astype(int)
        # mouth
        mean_mo = np.mean(landmarks[map_mo], 0)
        l_mo = np.max((np.max(np.max(landmarks[map_mo], 0) - np.min(landmarks[map_mo], 0)) / 2, 16))
        location_mo = np.hstack((mean_mo - l_mo + 1, mean_mo + l_mo)).astype(int)
    except:
        return 0
    return torch.from_numpy(location_le).unsqueeze(0), torch.from_numpy(location_re).unsqueeze(0), torch.from_numpy(
        location_no).unsqueeze(0), torch.from_numpy(location_mo).unsqueeze(0)


def obtain_inputs(img_path, landmark_path, name):
    a_paths = os.path.join(img_path, name)
    a = Image.open(a_paths).convert('RGB')
    part_locations = get_part_location(landmark_path, name)
    if part_locations == 0:
        return 0
    c = a
    a = add_up_sample(a)
    a = transforms.ToTensor()(a)
    c = transforms.ToTensor()(c)
    a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
    c = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(c)
    return {'A': a.unsqueeze(0), 'C': c.unsqueeze(0), 'A_paths': a_paths, 'Part_locations': part_locations}


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.which_epoch = 'latest'  #

    #######################################################################
    ########################### Test Param ################################
    #######################################################################
    opt.gpu_ids = []  # gpu id. if use cpu, set opt.gpu_ids = []
    # TestImgPath = './TestData/TestWhole' # test image path
    # ResultsDir = './Results/TestWholeResults' #save path
    # UpScaleWhole = 4  # the upsamle scale. It should be noted that our face results are fixed to 512.
    TestImgPath = opt.test_path
    ResultsDir = opt.results_dir
    UpScaleWhole = opt.upscale_factor

    print('\n###################### Now Running the X {} task ##############################'.format(UpScaleWhole))

    #######################################################################
    ###########Step 1: Crop and Align Face from the whole Image ###########
    #######################################################################
    print('\n###############################################################################')
    print('####################### Step 1: Crop and Align Face ###########################')
    print('###############################################################################\n')

    detector = dlib.cnn_face_detection_model_v1('./packages/mmod_human_face_detector.dat')
    sp = dlib.shape_predictor('./packages/shape_predictor_5_face_landmarks.dat')
    reference = np.load('./packages/FFHQ_template.npy') / 2
    SaveInputPath = os.path.join(ResultsDir, 'Step0_Input')
    if not os.path.exists(SaveInputPath):
        os.makedirs(SaveInputPath)
    SaveCropPath = os.path.join(ResultsDir, 'Step1_CropImg')
    if not os.path.exists(SaveCropPath):
        os.makedirs(SaveCropPath)

    SaveParamPath = os.path.join(ResultsDir, 'Step1_AffineParam')  # save the inverse affine parameters
    if not os.path.exists(SaveParamPath):
        os.makedirs(SaveParamPath)

    ImgPaths = make_dataset(TestImgPath)
    for i, ImgPath in enumerate(ImgPaths):
        img_name = os.path.split(ImgPath)[-1]
        print('Crop and Align {} image'.format(img_name))
        SavePath = os.path.join(SaveCropPath, img_name)
        SaveInput = os.path.join(SaveInputPath, img_name)
        SaveParam = os.path.join(SaveParamPath, img_name + '.npy')
        align_and_save(ImgPath, SavePath, SaveInput, SaveParam, UpScaleWhole)

    #######################################################################
    ####### Step 2: Face Landmark Detection from the Cropped Image ########
    #######################################################################
    print('\n###############################################################################')
    print('####################### Step 2: Face Landmark Detection #######################')
    print('###############################################################################\n')

    SaveLandmarkPath = os.path.join(ResultsDir, 'Step2_landmarks')
    if len(opt.gpu_ids) > 0:
        dev = 'cuda:{}'.format(opt.gpu_ids[0])
    else:
        dev = 'cpu'
    FD = face_alignment.FaceAlignment(face_alignment.landmarksType._2D, device=dev, flip_input=False)
    if not os.path.exists(SaveLandmarkPath):
        os.makedirs(SaveLandmarkPath)
    ImgPaths = make_dataset(SaveCropPath)
    for i, ImgPath in enumerate(ImgPaths):
        img_name = os.path.split(ImgPath)[-1]
        print('Detecting {}'.format(img_name))
        Img = io.imread(ImgPath)
        try:
            PredsAll = FD.get_landmarks(Img)
        except:
            print('\t################ Error in face detection, continue...')
            continue
        if PredsAll is None:
            print('\t################ No face, continue...')
            continue
        ins = 0
        if len(PredsAll) != 1:
            hights = []
            for l in PredsAll:
                hights.append(l[8, 1] - l[19, 1])
            ins = hights.index(max(hights))
            # print('\t################ Warning: Detected too many face, only handle the largest one...')
            # continue
        preds = PredsAll[ins]
        AddLength = np.sqrt(np.sum(np.power(preds[27][0:2] - preds[33][0:2], 2)))
        SaveName = img_name + '.txt'
        np.savetxt(os.path.join(SaveLandmarkPath, SaveName), preds[:, 0:2], fmt='%.3f')

    #######################################################################
    ####################### Step 3: Face Restoration ######################
    #######################################################################

    print('\n###############################################################################')
    print('####################### Step 3: Face Restoration ##############################')
    print('###############################################################################\n')

    SaveRestorePath = os.path.join(ResultsDir, 'Step3_RestoreCropFace')  # Only Face Results
    if not os.path.exists(SaveRestorePath):
        os.makedirs(SaveRestorePath)
    model = create_model(opt)
    model.setup(opt)
    # test
    ImgPaths = make_dataset(SaveCropPath)
    total = 0
    for i, ImgPath in enumerate(ImgPaths):
        img_name = os.path.split(ImgPath)[-1]
        print('Restoring {}'.format(img_name))
        torch.cuda.empty_cache()
        data = obtain_inputs(SaveCropPath, SaveLandmarkPath, img_name)
        if data == 0:
            print('\t################ Error in landmark file, continue...')
            continue  #
        total = total + 1
        model.set_input(data)
        try:
            model.test()
            visuals = model.get_current_visuals()
            save_crop(visuals, os.path.join(SaveRestorePath, img_name))
        except Exception as e:
            print('\t################ Error in enhancing this image: {}'.format(str(e)))
            print('\t################ continue...')
            continue

    #######################################################################
    ############ Step 4: Paste the Results to the Input Image #############
    #######################################################################

    print('\n###############################################################################')
    print('############### Step 4: Paste the Restored Face to the Input Image ############')
    print('###############################################################################\n')

    SaveFianlPath = os.path.join(ResultsDir, 'Step4_FinalResults')
    if not os.path.exists(SaveFianlPath):
        os.makedirs(SaveFianlPath)
    ImgPaths = make_dataset(SaveRestorePath)
    for i, ImgPath in enumerate(ImgPaths):
        img_name = os.path.split(ImgPath)[-1]
        print('Final Restoring {}'.format(img_name))
        WholeInputPath = os.path.join(TestImgPath, img_name)
        FaceResultPath = os.path.join(SaveRestorePath, img_name)
        ParamPath = os.path.join(SaveParamPath, img_name + '.npy')
        SaveWholePath = os.path.join(SaveFianlPath, img_name)
        reverse_align(WholeInputPath, FaceResultPath, ParamPath, SaveWholePath, UpScaleWhole)

    print('\nAll results are saved in {} \n'.format(ResultsDir))
