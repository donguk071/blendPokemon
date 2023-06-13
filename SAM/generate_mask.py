import argparse
import os

import cv2 as cv
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

# 명령행 인수를 파싱하기 위한 ArgumentParser 객체 생성
parser = argparse.ArgumentParser()

# --input 옵션: 입력 이미지 또는 이미지 폴더의 경로를 지정합니다.
parser.add_argument(
    "--input",
    type=str,
    default="samples",
    help="Path to either a single input image or folder of images.",
)

# --output 옵션: 마스크가 출력될 디렉토리 경로를 지정합니다.
parser.add_argument(
    "--output",
    type=str,
    default="samples/outputs",
    help=("Path to the directory where masks will be output."),
)

# --model-type 옵션: 사용할 모델 유형을 선택합니다. ['default', 'vit_h', 'vit_l', 'vit_b'] 중에서 선택합니다.
parser.add_argument(
    "--model-type",
    type=str,
    default="vit_h",
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

# --checkpoint 옵션: 마스크 생성에 사용할 SAM 체크포인트 파일의 경로를 지정합니다.
parser.add_argument(
    "--checkpoint",
    type=str,
    default="sam_vit_h_4b8939.pth",
    help="The path to the SAM checkpoint to use for mask generation.",
)

# --device 옵션: 실행할 디바이스를 지정합니다. (기본값: 'cuda')
parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")


# 이미지의 마스크를 생성하는 함수
def generate_mask(
    image_path: str,
    sam_checkpoint: str = "../SAM/sam_vit_h_4b8939.pth",
    model_type: str = "vit_h",
    device: str = "cuda",
) -> None:
    # 모델 로딩
    print("Loading model...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # 출력 디렉토리 생성
    output = os.path.join(os.path.dirname(image_path), "outputs")
    os.makedirs(output, exist_ok=True)

    # 이미지 로딩
    print(f"Processing '{image_path}'...")
    image = cv.imread(image_path)

    # 이미지를 RGB로 변환하여 모델에 입력
    predictor.set_image(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    # 마스크 결과를 저장할 디렉토리 설정
    base = os.path.basename(image_path)
    base = os.path.splitext(base)[0]
    save_base = os.path.join(output, base)

    # 마우스 콜백 함수 정의
    def mouse_callback(event, x, y, flags, param):
        global mask, mask_inv, img_fg, img_bg

        if event == cv.EVENT_LBUTTONDOWN:
            # 마우스 클릭한 위치를 좌표로 변환하여 입력으로 사용
            input_point = np.array([[x, y]])
            input_label = np.array([1])

            # 모델을 사용하여 마스크 생성
            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

            # 마스크 전처리 및 결과 생성
            mask = masks[-1]
            h, w = mask.shape[-2:]
            mask = mask.reshape(h, w, 1).astype(np.uint8) * 255
            mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            mask_inv = cv.bitwise_not(mask)

            img_fg = cv.bitwise_and(image, mask)
            img_bg = cv.bitwise_and(image, mask_inv)

            img_result = cv.vconcat([cv.hconcat([mask, mask_inv]), cv.hconcat([img_fg, img_bg])])
            cv.imshow("res", img_result)

    # 이미지 윈도우 생성 및 마우스 콜백 함수 연결
    winname = "image"
    cv.namedWindow(winname)
    cv.setMouseCallback(winname, mouse_callback)

    # 이미지 출력 및 마스크 확인 반복
    while True:
        cv.imshow(winname, image)
        k = cv.waitKey(0)
        if k == 27:  # ESC 키를 눌러서 종료
            break
        elif k == ord("s"):  # 's' 키를 눌러서 저장하고 종료
            os.makedirs(save_base, exist_ok=True)
            cv.imwrite(os.path.join(save_base, "mask.png"), mask)
            cv.imwrite(os.path.join(save_base, "mask_inv.png"), mask_inv)
            cv.imwrite(os.path.join(save_base, "img_fg.png"), img_fg)
            cv.imwrite(os.path.join(save_base, "img_bg.png"), img_bg)
            break
    cv.destroyAllWindows()
    print("Done!")


def main(args: argparse.Namespace) -> None:
    # 모델 로딩
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)

    # 입력이 폴더가 아니면 리스트에 하나의 요소로 추가
    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        # 입력이 폴더면 해당 폴더의 파일 목록을 가져옴
        targets = [f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))]
        targets = [os.path.join(args.input, f) for f in targets]

    for t in targets:
        # 이미지 로딩
        print(f"Processing '{t}'...")
        image = cv.imread(t)

        # 이미지를 RGB로 변환하여 모델에 입력
        predictor.set_image(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        # 마스크 결과를 저장할 디렉토리 생성
        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)

        # 마우스 콜백 함수 정의
        def mouse_callback(event, x, y, flags, param):
            global mask, mask_inv, img_fg, img_bg

            if event == cv.EVENT_LBUTTONDOWN:
                # 마우스 클릭한 위치를 좌표로 변환하여 입력으로 사용
                input_point = np.array([[x, y]])
                input_label = np.array([1])

                # 모델을 사용하여 마스크 생성
                masks, _, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )

                # 마스크 전처리 및 결과 생성
                mask = masks[-1]
                h, w = mask.shape[-2:]
                mask = mask.reshape(h, w, 1).astype(np.uint8) * 255
                mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
                mask_inv = cv.bitwise_not(mask)

                img_fg = cv.bitwise_and(image, mask)
                img_bg = cv.bitwise_and(image, mask_inv)

                img_result = cv.vconcat([cv.hconcat([mask, mask_inv]), cv.hconcat([img_fg, img_bg])])
                cv.imshow("res", img_result)

        # 이미지 윈도우 생성 및 마우스 콜백 함수 연결
        winname = "image"
        cv.namedWindow(winname)
        cv.setMouseCallback(winname, mouse_callback)

        # 이미지 출력 및 마스크 확인 반복
        while True:
            cv.imshow(winname, image)
            k = cv.waitKey(0)
            if k == 27:  # ESC 키를 눌러서 종료
                break
            elif k == ord("s"):  # 's' 키를 눌러서 저장하고 종료
                os.makedirs(save_base, exist_ok=True)
                cv.imwrite(os.path.join(save_base, "mask.png"), mask)
                cv.imwrite(os.path.join(save_base, "mask_inv.png"), mask_inv)
                cv.imwrite(os.path.join(save_base, "img_fg.png"), img_fg)
                cv.imwrite(os.path.join(save_base, "img_bg.png"), img_bg)
                break

        cv.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
