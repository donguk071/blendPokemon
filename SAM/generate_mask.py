import argparse
import os

import cv2 as cv
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    type=str,
    default="samples",
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    default="samples/outputs",
    help=("Path to the directory where masks will be output."),
)

parser.add_argument(
    "--model-type",
    type=str,
    default="vit_h",
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default="sam_vit_h_4b8939.pth",
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")


def generate_mask(
    image_path: str,
    sam_checkpoint: str = "../SAM/sam_vit_h_4b8939.pth",
    model_type: str = "vit_h",
    device: str = "cuda",
) -> None:
    print("Loading model...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    output = os.path.join(os.path.dirname(image_path), "outputs")
    os.makedirs(output, exist_ok=True)

    print(f"Processing '{image_path}'...")
    image = cv.imread(image_path)

    predictor.set_image(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    base = os.path.basename(image_path)
    base = os.path.splitext(base)[0]
    save_base = os.path.join(output, base)

    def mouse_callback(event, x, y, flags, param):
        global mask, mask_inv, img_fg, img_bg

        if event == cv.EVENT_LBUTTONDOWN:
            input_point = np.array([[x, y]])
            input_label = np.array([1])

            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

            mask = masks[-1]
            h, w = mask.shape[-2:]
            mask = mask.reshape(h, w, 1).astype(np.uint8) * 255
            mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            mask_inv = cv.bitwise_not(mask)

            img_fg = cv.bitwise_and(image, mask)
            img_bg = cv.bitwise_and(image, mask_inv)

            img_result = cv.vconcat([cv.hconcat([mask, mask_inv]), cv.hconcat([img_fg, img_bg])])
            cv.imshow("res", img_result)

    winname = "image"
    cv.namedWindow(winname)
    cv.setMouseCallback(winname, mouse_callback)

    while True:
        cv.imshow(winname, image)
        k = cv.waitKey(0)
        if k == 27:  # wait for ESC key to exit
            break
        elif k == ord("s"):  # wait for 's' key to save and exit
            os.makedirs(save_base, exist_ok=True)
            cv.imwrite(os.path.join(save_base, "mask.png"), mask)
            cv.imwrite(os.path.join(save_base, "mask_inv.png"), mask_inv)
            cv.imwrite(os.path.join(save_base, "img_fg.png"), img_fg)
            cv.imwrite(os.path.join(save_base, "img_bg.png"), img_bg)
            break
    cv.destroyAllWindows()
    print("Done!")


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    os.makedirs(args.output, exist_ok=True)

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))]
        targets = [os.path.join(args.input, f) for f in targets]

    for t in targets:
        print(f"Processing '{t}'...")
        image = cv.imread(t)

        predictor.set_image(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)

        def mouse_callback(event, x, y, flags, param):
            global mask, mask_inv, img_fg, img_bg

            if event == cv.EVENT_LBUTTONDOWN:
                input_point = np.array([[x, y]])
                input_label = np.array([1])

                masks, _, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )

                mask = masks[-1]
                h, w = mask.shape[-2:]
                mask = mask.reshape(h, w, 1).astype(np.uint8) * 255
                mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
                mask_inv = cv.bitwise_not(mask)

                img_fg = cv.bitwise_and(image, mask)
                img_bg = cv.bitwise_and(image, mask_inv)

                img_result = cv.vconcat([cv.hconcat([mask, mask_inv]), cv.hconcat([img_fg, img_bg])])
                cv.imshow("res", img_result)

        winname = "image"
        cv.namedWindow(winname)
        cv.setMouseCallback(winname, mouse_callback)

        while True:
            cv.imshow(winname, image)
            k = cv.waitKey(0)
            if k == 27:  # wait for ESC key to exit
                break
            elif k == ord("s"):  # wait for 's' key to save and exit
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
