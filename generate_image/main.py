import sys

sys.path.append("..")

import argparse
import os

import cv2 as cv
import numpy as np

import LIIF
import MAT
import SAM

parser = argparse.ArgumentParser()

parser.add_argument("--img-bg", type=str, required=True, help="Path to a background image.")

parser.add_argument("--img-fg", type=str, required=True, help="Path to a foreground image.")


is_mouse_pressed = False
ix, iy = -1, -1
warp_mat = np.eye(3, 3)


def main(args: argparse.Namespace) -> None:
    SAM.generate_mask.generate_mask(args.img_bg)
    SAM.generate_mask.generate_mask(args.img_fg)

    output = os.path.join(os.path.dirname(args.img_bg), "outputs")

    base = os.path.basename(args.img_bg)
    base = os.path.splitext(base)[0]
    save_base = os.path.join(output, base)

    image = cv.imread(args.img_bg)
    image = cv.resize(image, (512, 512))

    os.makedirs("images", exist_ok=True)
    cv.imwrite(f"images/{base}.png", image)

    mask = cv.imread(os.path.join(save_base, "mask_inv.png"))
    mask = cv.resize(mask, (512, 512))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    mask = cv.erode(mask, kernel)

    os.makedirs("masks", exist_ok=True)
    cv.imwrite(f"masks/{base}.png", mask)

    MAT.module_inpainting.generate_images_ours(
        dpath="images",
        mpath="masks",
        outdir=os.path.join(output, "mat"),
    )

    img_bg = cv.imread(save_base + ".png")

    base = os.path.basename(args.img_fg)
    base = os.path.splitext(base)[0]
    save_base = os.path.join(output, base)

    img_fg = cv.imread(os.path.join(save_base, "img_fg.png"))
    img_fg = cv.resize(img_fg, (512, 512))

    mask = cv.imread(os.path.join(save_base, "mask_inv.png"))
    mask = cv.resize(mask, (512, 512))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.dilate(mask, kernel)

    def mouse_callback(event, x, y, flags, param):
        global warp_mat, ix, iy, is_mouse_pressed

        if event == cv.EVENT_LBUTTONDOWN:
            is_mouse_pressed = True
            ix, iy = x, y

        elif event == cv.EVENT_MOUSEMOVE:
            if is_mouse_pressed:
                tx, ty = x - ix, y - iy
                warp_mat[0, 2] += tx
                warp_mat[1, 2] += ty

                cv.warpAffine(img_fg, warp_mat[:2, :], (img_fg_dst.shape[1], img_fg_dst.shape[0]), img_fg_dst)
                cv.warpAffine(
                    src=mask,
                    M=warp_mat[:2, :],
                    dsize=(mask_dst.shape[1], mask_dst.shape[0]),
                    dst=mask_dst,
                    borderValue=(255, 255, 255),
                )

                cv.bitwise_and(img_bg, mask_dst, img_dst)
                cv.add(img_dst, img_fg_dst, img_dst)

                ix, iy = x, y

        elif event == cv.EVENT_LBUTTONUP:
            is_mouse_pressed = False

            tx, ty = x - ix, y - iy
            warp_mat[0, 2] += tx
            warp_mat[1, 2] += ty

            cv.warpAffine(img_fg, warp_mat[:2, :], (img_fg_dst.shape[1], img_fg_dst.shape[0]), img_fg_dst)
            cv.warpAffine(
                src=mask,
                M=warp_mat[:2, :],
                dsize=(mask_dst.shape[1], mask_dst.shape[0]),
                dst=mask_dst,
                borderValue=(255, 255, 255),
            )

            cv.bitwise_and(img_bg, mask_dst, img_dst)
            cv.add(img_dst, img_fg_dst, img_dst)

        elif event == cv.EVENT_MOUSEWHEEL:
            if flags > 0:
                scale = 1.1
            else:
                scale = 0.9

            translate_mat = np.eye(3, 3)
            translate_mat[0, 2] = -x
            translate_mat[1, 2] = -y

            scale_mat = np.eye(3, 3)
            scale_mat[0, 0] = scale
            scale_mat[1, 1] = scale

            translate_mat_inv = np.linalg.inv(translate_mat)

            warp_mat = translate_mat_inv @ scale_mat @ translate_mat @ warp_mat

            cv.warpAffine(img_fg, warp_mat[:2, :], (img_fg_dst.shape[1], img_fg_dst.shape[0]), img_fg_dst)
            cv.warpAffine(
                src=mask,
                M=warp_mat[:2, :],
                dsize=(mask_dst.shape[1], mask_dst.shape[0]),
                dst=mask_dst,
                borderValue=(255, 255, 255),
            )

            cv.bitwise_and(img_bg, mask_dst, img_dst)
            cv.add(img_dst, img_fg_dst, img_dst)

    img_fg_dst = img_fg.copy()
    mask_dst = mask.copy()

    img_dst = cv.bitwise_and(img_bg, mask)
    img_dst = cv.add(img_dst, img_fg)

    winname = "output"
    cv.namedWindow(winname)
    cv.setMouseCallback(winname, mouse_callback)

    filename = "output.png"

    while True:
        cv.imshow(winname, img_dst)
        k = cv.waitKey(1)
        if k == 27:  # wait for ESC key to exit
            break
        elif k == ord("s"):  # wait for 's' key to save and exit
            cv.imwrite(filename, img_dst)
            break

    cv.destroyAllWindows()

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
