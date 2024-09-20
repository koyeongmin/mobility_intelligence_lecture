import cv2
import numpy as np
from copy import deepcopy

def crop_quarter(image, height, width):
    return image[0:int(height/2), 0:int(width/2)] #절반만 자르고 나머지는 버림

def resize_to_original(image, original_height, original_width):
    return cv2.resize(image, (original_width, original_height))

def add_gaussian(image, height, width):
    gaussian_noise = np.random.normal(0.1, 0.5, (height, width, 3)) #가우시안 노이즈 생성
    result = image + gaussian_noise #노이즈 더하기
    result = result.astype(np.uint8) #노이즈가 실수형이므로, 정수형으로 변환
    result = np.clip(result, 0, 255) #이미지는 0~255사이의 값을 가져야하므로, 범위를 넘어서는 값 처리
    return result

def main(args=None):

    # 이미지 불러오기
    file_name = "camera.png" # 파일이름
    img = cv2.imread(file_name) # 파일 불러오기
    original_image = deepcopy(img) # 원본 이미지 보관
    cv2.imshow("original", img) # 불러온 이미지파일을 새로운 창에서 열기
    #cv2.waitKey(0) # 대기(프로그램이 끝나지 않도록, 0일때는 새로운 키입력이 있을때 까지 대기, 다른 숫자는 그 수 만큼(초) 대기)

    # 이미지 크기 정보 얻기
    print(img.shape)  # (height, width, channel)
    height, width, _ = img.shape 

    # 이미지 1/4로 자르기
    img = crop_quarter(img, height, width)
    #cv2.imshow("crop", img)
    #cv2.waitKey(0)

    # 다시 원래 크기로 resizing, 하지만 위에서 나머지 3/4는 버렸으므로 1/4짜리가 원래 크기로 확대됨
    img = resize_to_original(img, height, width)
    #cv2.imshow("zoom", img)
    #cv2.waitKey(0)

    # 가우시안 노이즈 추가
    img = add_gaussian(original_image, height, width)
    #cv2.imshow("Gaussian", img)
    #cv2.waitKey(0)

    # edge 필터
    cropped_image = original_image[int(height/2):height, :] #ROI
    canny_image = cv2.Canny(cropped_image, 100, 200, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(canny_image, 1, np.pi/180, 140, minLineLength=1, maxLineGap=10)
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
