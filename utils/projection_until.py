import os, sys, inspect, time
import torch
import numpy as np
import torchvision
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from utils.image_util import load_depth_label_pose

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# train on the GPU or on the CPU, if a GPU is not available
if torch.cuda.is_available():
    print("Using GPU to train")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ENET_TYPES = {'scannet': (41, [0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129])}  #classes, color mean/std
color_mean = [0.496342, 0.466664, 0.440796]
color_std = [0.277856, 0.28623, 0.291129]
# create camera intrinsics
input_image_dims = [320, 240]
# enet
# proj_image_dims = [40, 30]
# mask r cnn
proj_image_dims = [34, 25]

# 2d mask r cnn model
# get the model using our helper function
def get_model_instance_segmentation(num_classes=18):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=num_classes)
    return model


def read_lines_from_file(filename):
    assert os.path.isfile(filename)
    lines = open(filename).read().splitlines()
    return lines



def get_features_for_projection_multi(imagePaths, device, model_maskrcnn):
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    # these transform parameters are from source code of Mask R CNN
    transform = GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=image_mean, image_std=image_std)
    images = [Image.open(imagePath) for imagePath in imagePaths]
    image_tensors = [TF.to_tensor(image) for image in images]
    # let it be in list (can be multiple)
    results = []
    with torch.no_grad():
        for tensor in image_tensors:
            images, _ = transform([tensor])
            features = model_maskrcnn.backbone(images.tensors.to(device))
            features_to_be_projected = features[3]
            results.append(features_to_be_projected[0])
    return results

def get_features_enet_multi(imagePaths, device,model2d_fixed,model2d_trainable):
    images = [Image.open(imagePath) for imagePath in imagePaths]
    image_tensors = [TF.to_tensor(image) for image in images]
    # let it be in list (can be multiple)
    results = []
    with torch.no_grad():
        for tensor in image_tensors:
            tensor = torch.unsqueeze(tensor,0)
            imageft_fixed = model2d_fixed(tensor.to(device))
            imageft = model2d_trainable(imageft_fixed)
            imageft = imageft.cuda()
            results.append(imageft[0])
    return results


def load_frames_multi_2(data_path, image_names, depth_image_torch, color_image_torch, camera_pose, color_mean, color_std):
    color_images = []
    depth_images = []
    camera_poses = []
    for image_name in image_names:
        depth_file = os.path.join(data_path, 'depth', image_name+".png")
        color_file = os.path.join(data_path, 'color', image_name+".jpg")
        pose_file = os.path.join(data_path,  'pose', image_name+".txt")
        depth_image_dims = [depth_image_torch.shape[2], depth_image_torch.shape[1]]
        color_image_dims = [color_image_torch.shape[3], color_image_torch.shape[2]]
        normalize = transforms.Normalize(mean=color_mean, std=color_std)
        # load data
        depth_img, color_img, pose = load_depth_label_pose(depth_file, color_file, pose_file, depth_image_dims, color_image_dims, normalize)
        color_image = color_img
        depth_image = torch.from_numpy(depth_img)
        camera_pose = pose
        color_images.append(color_image)
        depth_images.append(depth_image)
        camera_poses.append(camera_pose)
    return color_images,depth_images,camera_poses


def project_images(data_path, image_names, mesh_vertices, args, model2d_fixed, model2d_trainable,projection, maskrcnn_model):
    depth_image = torch.cuda.FloatTensor(len(image_names), proj_image_dims[1], proj_image_dims[0])
    color_image = torch.cuda.FloatTensor(len(image_names), 3, input_image_dims[1], input_image_dims[0])
    camera_pose = torch.cuda.FloatTensor(len(image_names), 4, 4)
    color_images,depth_images,camera_poses = load_frames_multi_2(data_path, image_names, depth_image, color_image, camera_pose, color_mean, color_std)
    depth_images = torch.stack(depth_images)
    camera_poses = torch.stack(camera_poses)
    boundingmin, boundingmax, world_to_camera = projection.compute_projection_multi(camera_poses)
    lin_ind_volume = np.arange(0, mesh_vertices.shape[0],dtype=np.int)
    N = mesh_vertices.shape[0]
    mesh_vertices_original = mesh_vertices.copy()
    mesh_vertices_original = np.transpose(mesh_vertices_original)
    mesh_vertices_original = torch.from_numpy(mesh_vertices_original).float().to(device).cuda()
    mesh_vertices = mesh_vertices[:, 0:4]
    mesh_vertices = np.transpose(mesh_vertices)
    boundingmax = boundingmax.cpu().numpy()
    boundingmin = boundingmin.cpu().numpy()
    world_to_camera = world_to_camera.cuda()
    mesh_vertices = np.expand_dims(mesh_vertices, 0)
    mesh_vertices = np.repeat(mesh_vertices, len(image_names), axis=0)
    mask_frustum_bounds = np.greater_equal(mesh_vertices[:,0], np.expand_dims(boundingmin[:,0],1)) * np.greater_equal(mesh_vertices[:,1], np.expand_dims(boundingmin[:,1],1)) * np.greater_equal(mesh_vertices[:,2], np.expand_dims(boundingmin[:,2],1))
    mask_frustum_bounds = mask_frustum_bounds * np.less(mesh_vertices[:,0], np.expand_dims(boundingmax[:,0],1)) * np.less(mesh_vertices[:,1], np.expand_dims(boundingmax[:,1],1)) * np.less(mesh_vertices[:,2], np.expand_dims(boundingmax[:,2],1))
    if not mask_frustum_bounds.any():
        # print('error: nothing in frustum bounds')
        return None
    lin_ind_volume = np.expand_dims(lin_ind_volume, 0)
    lin_ind_volume = np.repeat(lin_ind_volume, len(image_names), axis=0)
    # lin_ind_volume = lin_ind_volume[mask_frustum_bounds]
    world_to_camera = world_to_camera.cpu().numpy()
    mesh_vertices = np.matmul(world_to_camera, mesh_vertices)
    # mesh_vertices = torch.bmm(world_to_camera, mesh_vertices)
    # transform to current frame
    mesh_vertices = np.moveaxis(mesh_vertices, 0, -2)
    # mesh_vertices = mesh_vertices.permute(1, 0, 2)
    # p = p[:,mask_frustum_bounds]
    # project into image
    mesh_vertices[0] = (mesh_vertices[0] * projection.intrinsic[0][0]) / mesh_vertices[2] + projection.intrinsic[0][2]
    mesh_vertices[1] = (mesh_vertices[1] * projection.intrinsic[1][1]) / mesh_vertices[2] + projection.intrinsic[1][2]
    pi = np.around(mesh_vertices).astype(np.int)
    valid_ind_mask = np.greater_equal(pi[0,:], 0) * np.greater_equal(pi[1,:], 0) * np.less(pi[0,:], proj_image_dims[0]) * np.less(pi[1,:], proj_image_dims[1])
    if not valid_ind_mask.any():
        # print('error: no valid image indices')
        return None
    valid_ind_mask = valid_ind_mask * mask_frustum_bounds
    pi = pi*valid_ind_mask.astype(np.int)
    image_ind_x = pi[0,:]
    image_ind_y = pi[1,:]
    image_ind_lin = image_ind_y * proj_image_dims[0] + image_ind_x
    depth = depth_images.detach().cpu().numpy()
    depth_vals = np.concatenate([np.expand_dims(np.take(a.reshape(-1),i,0),0) for a, i in zip(depth, image_ind_lin)])
    depth_mask = np.greater_equal(depth_vals, args.depth_min) * np.less_equal(depth_vals, args.depth_max) * np.less_equal(np.absolute(depth_vals - mesh_vertices[2]*valid_ind_mask.astype(float)), args.voxel_size)

    if not depth_mask.any():
        # print('error: no valid depths')
        return None
    final_mask = (valid_ind_mask*depth_mask)
    lin_indices_3d = [a[i] for a, i in zip(lin_ind_volume, final_mask)]
    lin_indices_2d = [np.take(a, np.argwhere(i)[:, 0], 0) for a, i in zip(image_ind_lin, final_mask)]
    # use enet features
    # features_to_add = get_features_enet_multi([os.path.join(data_path, 'color', image_name+".jpg") for image_name in image_names], device, model2d_fixed, model2d_trainable)
    # use mask r cnn features
    features_to_add = get_features_for_projection_multi([os.path.join(data_path, 'color', image_name+".jpg") for image_name in image_names], device, maskrcnn_model)
    num_label_ft = 1 if len(features_to_add[0].shape) == 2 else features_to_add[0].shape[0]
    output = features_to_add[0].new(4 + num_label_ft, N).fill_(0)
    output[0:4, :] = mesh_vertices_original[0:4, :]
    output = output.detach().cpu().numpy()
#    num_ind = lin_indices_3d[0]
#    if num_ind > 0:
    features_to_add = [feature.cpu().numpy() for feature in features_to_add]
    for feature, lin_index_2d, lin_index_3d in zip(features_to_add, lin_indices_2d, lin_indices_3d):
        vals = np.take(feature.reshape(num_label_ft, -1), lin_index_2d, 1)
        # TODO change type to cpu
        output.reshape(num_label_ft+4, -1)[4:, lin_index_3d] = vals
    output = np.transpose(output)
    return output

# ROOT_DIR:indoor-objects
def scannet_projection(vertices, intrinsic, projection, scene_id, args, model2d_fixed, model2d_trainable, maskrcnn_model):
    # load vertices
    mesh_vertices = vertices[:,0:3]
    # Load alignments
    lines = open(args.RAW_DATA_DIR + scene_id + '/' + scene_id + '.txt').readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                                 for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    #inverse
    inv_axis_align_matrix_T = np.linalg.inv(axis_align_matrix.T)
    # add addition ones to the end or each position in mesh_vertices
    mesh_vertices = np.append(mesh_vertices, np.ones((mesh_vertices.shape[0], 1)), axis=1)
    mesh_vertices = np.dot(mesh_vertices, inv_axis_align_matrix_T)
    # add zeros dimension
    mesh_vertices = np.concatenate((mesh_vertices[:], np.zeros(((mesh_vertices.shape[0], 256)))), axis=1)
    # load_images
    image_path = os.path.join(args.data_path_2d, scene_id, 'color')
    images = []

    for image_name in os.listdir(image_path):
        image_name = image_name.replace(".jpg", "", 1)
        images.append(image_name)

    interval = round(len(images) / args.num_nearest_images)
    if interval == 0:
        interval = 1
    indices = np.arange(0, len(images), interval)
    indices = indices[:args.num_nearest_images]
    indices = indices.astype('int32')
    indices = list(indices)
    data_path = os.path.join(args.data_path_2d,scene_id)
    images = [images[i] for i in indices]
    with torch.no_grad():
        mesh_vertices = project_images(data_path, images, mesh_vertices,args, model2d_fixed, model2d_trainable,projection, maskrcnn_model)


    return mesh_vertices[:,4:]





