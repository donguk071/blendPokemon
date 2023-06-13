import sys

sys.path.append("..")  # 상위 디렉토리를 sys.path에 추가

import argparse
import os

import cv2 as cv
import numpy as np

import LIIF
import MAT
import SAM

# 명령행 인수를 파싱하기 위한 ArgumentParser 객체 생성
parser = argparse.ArgumentParser()

# 배경 이미지의 경로를 지정하는 명령행 인수
parser.add_argument("--img-bg", type=str, required=True, help="Path to a background image.")

# 전경 이미지의 경로를 지정하는 명령행 인수
parser.add_argument("--img-fg", type=str, required=True, help="Path to a foreground image.")


is_mouse_pressed = False  # 마우스 버튼 눌림 여부
ix, iy = -1, -1  # 마우스 클릭 위치
warp_mat = np.eye(3, 3)  # 변환 행렬 초기화


def main(args: argparse.Namespace) -> None:
    # 배경 이미지와 전경 이미지의 마스크 생성
    SAM.generate_mask.generate_mask(args.img_bg)
    SAM.generate_mask.generate_mask(args.img_fg)

    # 출력 디렉토리 설정
    output = os.path.join(os.path.dirname(args.img_bg), "outputs")

    base = os.path.basename(args.img_bg)
    base = os.path.splitext(base)[0]
    save_base = os.path.join(output, base)

    # 배경 이미지 로드 및 크기 조정
    image = cv.imread(args.img_bg)
    image = cv.resize(image, (512, 512))

    # 이미지 저장
    os.makedirs("images", exist_ok=True)
    cv.imwrite(f"images/{base}.png", image)

    # 배경 이미지의 마스크 로드 및 전처리
    mask = cv.imread(os.path.join(save_base, "mask_inv.png"))
    mask = cv.resize(mask, (512, 512))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    mask = cv.erode(mask, kernel)

    # 마스크 저장
    os.makedirs("masks", exist_ok=True)
    cv.imwrite(f"masks/{base}.png", mask)

    # MAT 모듈을 사용하여 이미지 복원
    MAT.module_inpainting.generate_images_ours(
        dpath="images",
        mpath="masks",
        outdir=os.path.join(output, "mat"),
    )

    # 복원된 배경 이미지 로드
    img_bg = cv.imread(os.path.join(output, "mat", f"{base}.png"))

    base = os.path.basename(args.img_fg)
    base = os.path.splitext(base)[0]
    save_base = os.path.join(output, base)

    # 전경 이미지 로드 및 크기 조정
    img_fg = cv.imread(os.path.join(save_base, "img_fg.png"))
    img_fg = cv.resize(img_fg, (512, 512))

    # 전경 이미지의 마스크 로드 및 전처리
    mask = cv.imread(os.path.join(save_base, "mask_inv.png"))
    mask = cv.resize(mask, (512, 512))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.dilate(mask, kernel)

    # 마우스 콜백 함수 정의
    def mouse_callback(event, x, y, flags, param):
        global warp_mat, ix, iy, is_mouse_pressed

        # 마우스 왼쪽 버튼을 누르는 이벤트 처리
        if event == cv.EVENT_LBUTTONDOWN:
            is_mouse_pressed = True
            ix, iy = x, y

        # 마우스 이동 이벤트 처리
        elif event == cv.EVENT_MOUSEMOVE:
            if is_mouse_pressed:
                # 이동 거리 계산
                tx, ty = x - ix, y - iy

                # 이동 거리에 따라 변환 행렬 업데이트
                warp_mat[0, 2] += tx
                warp_mat[1, 2] += ty

                # 전경 이미지 및 마스크 이동
                cv.warpAffine(img_fg, warp_mat[:2, :], (img_fg_dst.shape[1], img_fg_dst.shape[0]), img_fg_dst)
                cv.warpAffine(
                    src=mask,
                    M=warp_mat[:2, :],
                    dsize=(mask_dst.shape[1], mask_dst.shape[0]),
                    dst=mask_dst,
                    borderValue=(255, 255, 255),
                )

                # 합성된 이미지 생성
                cv.bitwise_and(img_bg, mask_dst, img_dst)
                cv.add(img_dst, img_fg_dst, img_dst)

                ix, iy = x, y

        # 마우스 왼쪽 버튼을 뗀 이벤트 처리
        elif event == cv.EVENT_LBUTTONUP:
            is_mouse_pressed = False

            # 이동 거리 계산
            tx, ty = x - ix, y - iy

            # 이동 거리에 따라 변환 행렬 업데이트
            warp_mat[0, 2] += tx
            warp_mat[1, 2] += ty

            # 전경 이미지 및 마스크 이동
            cv.warpAffine(img_fg, warp_mat[:2, :], (img_fg_dst.shape[1], img_fg_dst.shape[0]), img_fg_dst)
            cv.warpAffine(
                src=mask,
                M=warp_mat[:2, :],
                dsize=(mask_dst.shape[1], mask_dst.shape[0]),
                dst=mask_dst,
                borderValue=(255, 255, 255),
            )

            # 합성된 이미지 생성
            cv.bitwise_and(img_bg, mask_dst, img_dst)
            cv.add(img_dst, img_fg_dst, img_dst)

        # 마우스 휠 이벤트 처리
        elif event == cv.EVENT_MOUSEWHEEL:
            if flags > 0:
                scale = 1.1  # 휠을 위로 스크롤 할 경우 이미지 크기 확대
            else:
                scale = 0.9  # 휠을 아래로 스크롤 할 경우 이미지 크기 축소

            # 이미지 이동을 위한 변환 행렬 계산
            translate_mat = np.eye(3, 3)
            translate_mat[0, 2] = -x
            translate_mat[1, 2] = -y

            # 이미지 크기 조정을 위한 변환 행렬 계산
            scale_mat = np.eye(3, 3)
            scale_mat[0, 0] = scale
            scale_mat[1, 1] = scale

            translate_mat_inv = np.linalg.inv(translate_mat)

            # 변환 행렬 계산
            warp_mat = translate_mat_inv @ scale_mat @ translate_mat @ warp_mat

            # 전경 이미지 및 마스크 크기 조정
            cv.warpAffine(img_fg, warp_mat[:2, :], (img_fg_dst.shape[1], img_fg_dst.shape[0]), img_fg_dst)
            cv.warpAffine(
                src=mask,
                M=warp_mat[:2, :],
                dsize=(mask_dst.shape[1], mask_dst.shape[0]),
                dst=mask_dst,
                borderValue=(255, 255, 255),
            )

            # 합성된 이미지 생성
            cv.bitwise_and(img_bg, mask_dst, img_dst)
            cv.add(img_dst, img_fg_dst, img_dst)

    img_fg_dst = img_fg.copy()
    mask_dst = mask.copy()

    img_dst = cv.bitwise_and(img_bg, mask)
    img_dst = cv.add(img_dst, img_fg)

    # 출력 윈도우 생성 및 마우스 콜백 함수 연결
    winname = "output"
    cv.namedWindow(winname)
    cv.setMouseCallback(winname, mouse_callback)

    filename = "output.png"

    # 이미지 출력 반복
    while True:
        cv.imshow(winname, img_dst)
        k = cv.waitKey(1)
        if k == 27:  # ESC 키를 눌러서 종료
            break
        elif k == ord("s"):  # 's' 키를 눌러서 저장하고 종료
            cv.imwrite(filename, img_dst)
            break

    cv.destroyAllWindows()

    # LIIF 모듈을 사용하여 고해상도 이미지 생성
    LIIF.module_super_resolution.make_high_resolution(
        img_path=filename,
        output_path=filename,
        resolution_width=img_dst.shape[1] * 2,
        resolution_height=img_dst.shape[0] * 2,
    )

    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
