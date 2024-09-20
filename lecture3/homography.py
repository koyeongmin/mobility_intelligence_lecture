import cv2
import numpy as np
from copy import deepcopy

def main(args=None):

    # 이미지 불러오기
    file_name = "camera.png"
    img = cv2.imread(file_name)
    original_image = deepcopy(img)
    cv2.imshow("original", img) 

    src_list = np.float32([
        [488, 192],
        [580, 192],
        [142, 373],
        [820, 373]
    ])

    dst_list = np.float32([
        [0, 0],
        [480, 0],
        [0, 480],
        [480, 480]
    ])

    M = cv2.getPerspectiveTransform(src_list, dst_list)
    warp_img = cv2.warpPerspective(img, M, (480, 480), flags=cv2.INTER_LINEAR)
    cv2.imshow("warpped image", warp_img) 
    
    # edge 필터
    canny_image = cv2.Canny(warp_img, 100, 200, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(canny_image, 1, np.pi/180, 50, minLineLength=1, maxLineGap=100)
    hough_image = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for i in range(lines.shape[0]):
            pt1 = (lines[i][0][0], lines[i][0][1]) #시작점 좌표
            pt2 = (lines[i][0][2], lines[i][0][3]) #끝점 좌표
            cv2.line(hough_image, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("line detection", hough_image)
    cv2.waitKey(0)

    return 0

if __name__ == '__main__':
    main()