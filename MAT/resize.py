import cv2

# 원본 이미지 로드
image1 = cv2.imread('mytest/pockmon1.png')
image2 = cv2.imread('mytest/pockmon_mask1.png')

# 원하는 출력 크기 설정
width = 512
height = 512

# 이미지 크기 조정
resized_image1 = cv2.resize(image1, (width, height))
resized_image2 = cv2.resize(image2, (width, height))

# 조정된 이미지 저장
cv2.imwrite('test_sets/CelebA-HQ/images/test3.png', resized_image1)
cv2.imwrite('test_sets/CelebA-HQ/masks/mask3.png', resized_image2)

