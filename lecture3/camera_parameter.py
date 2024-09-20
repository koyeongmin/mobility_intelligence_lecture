import cv2
import numpy as np
import open3d as o3d
import PIL.Image as pil
import struct
import matplotlib.pyplot as plt

from copy import deepcopy


###########################################
# 포인트 클라우드 시각화
###########################################
def visualization_open3d(data): # show point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])
    o3d.visualization.draw_geometries([pcd])

###########################################
# 포인트 클라우드를 이미지에 투영 (가시성 개선 필요)
###########################################
def visualization_projection(image_file, data, x, y, z):
    img = cv2.imread(image_file)
    img_mapped = img.copy()
    img_h, img_w = img.shape[:2]

    max_depth = np.max(z)
    min_depth = np.min(z)
    for i, (ix, iy, iz) in enumerate(zip(x, y, z)):
        if 0 <= ix < img_w and 0 <= iy < img_h:
            c = (iz - min_depth) / (max_depth - min_depth + 0.000001)
            c = 255 - ((1-c) * 255)
            cv2.circle(img_mapped, (ix, iy), radius=1, color=(c, c, c), thickness=2)

    img_mapped_rgb = cv2.cvtColor(img_mapped, cv2.COLOR_BGR2RGB)

    plt.imshow(img_mapped_rgb)
    plt.show()

###########################################
# 뎁스 이미지에서 포인트 클라우드 얻기
###########################################
def generate_pointcloud(depth, K):

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 255)
    z = np.where(valid, depth, 0)
    x = np.where(valid, z * (c - K[0,2]) / K[0,0], 0)
    y = np.where(valid, z * (r - K[1,2]) / K[1,1], 0)
    return np.dstack((x, y, z)).reshape(-1,3)

def main(args=None):

    ###########################################
    # 파라미터
    ###########################################
    camera_file = "camera.png"
    depth_file = "depth.png"
    lidar_file = "lidar.bin"

    K = np.array([[0.58, 0, 0.5, 0],
                        [0, 1.92, 0.5, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32) # normalized intrinsic parameter
    
    R0_rect = np.array([[9.998817e-01, 1.511453e-02, -2.841595e-03],
                      [-1.511724e-02, 9.998853e-01, -9.338510e-04],
                      [2.827154e-03, 9.766976e-04, 9.999955e-01]])
    
    P = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
                 [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
                  [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]])
    
    velo_to_cam = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                            [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
                            [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01]])
    
    ###########################################
    # 파일 불러오기
    ###########################################
    img = cv2.imread(camera_file) # 카메라 이미지 파일 불러오기
    depth_png = np.array(pil.open(depth_file), dtype=int) # 깊이 이미지 파일 불러오기
    depth = depth_png.astype(np.float32)/256.0 # pixel value -> depth value

    height, width, _ = img.shape 
    K[0, :] *= width
    K[1, :] *= height # 내부 파라미터 조절

    # 라이다 데이터 확인
    with open(lidar_file, 'rb') as f: # show point cloud
        data = np.fromfile(f, dtype=np.float32).reshape(-1,4) # x, y, z, intensity
    visualization_open3d(data)

    ###########################################
    # 라이다 데이터를 이미지 위에 투영
    ###########################################
    R0 = np.eye(4)
    R0[:3, :3] = R0_rect #3x3 행렬인 R0_rect을 4x4 행렬로 변환

    with open(lidar_file, 'rb') as f: # 3d 데이터 로드
        point_cloud = np.fromfile(f, dtype=np.float32).reshape(-1,4)
        point_cloud = point_cloud[:, :3]
    point_cloud_homo = np.column_stack([point_cloud, np.ones((point_cloud.shape[0], 1))]) # 3d 데이터 -> homogeneous coordinate
    cam_point_cloud = np.dot(velo_to_cam, np.transpose(point_cloud_homo)) # 라이다 좌표계에서 카메라 좌표계로
    cam_point_cloud = cam_point_cloud.T
    cam_point_cloud_homo = np.column_stack([cam_point_cloud, np.ones((cam_point_cloud.shape[0], 1))]) # homogeneous coordinate

    p_r0 = np.dot(P, R0) # 투영 행렬 구하기
    p_r0_x = np.dot(p_r0, np.transpose(cam_point_cloud_homo)) # 카메라 좌표계로 변환된 3D 포인트에 투영행렬을 곱해서 이미지 좌표계로
    points_2d = np.transpose(p_r0_x) 

    z = points_2d[:, 2]
    x = (points_2d[:, 0] / z).astype(np.int32)[z>0] 
    y = (points_2d[:, 1] / z).astype(np.int32)[z>0]

    visualization_projection(camera_file, data, x, y, z) # show

    ###########################################
    # 뎁스 이미지에서 포인트 클라우드 얻기(이미지 투영 과정의 역변환)
    ###########################################
    inv_K = np.linalg.inv(K)
    print(depth)
    point = generate_pointcloud(depth, K)
    visualization_open3d(point)

if __name__ == '__main__':
    main()