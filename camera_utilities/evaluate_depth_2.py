"""RMS depth error vs. ground truth for a top-down D405 over a flat board.

Single depth frame. Depth units: 0.1 mm per raw unit (D405 default).

The board mask no longer requires an exact pattern size: it sweeps grid
sizes from large to small and uses the largest checkerboard grid found.
(findChessboardCornersSB is unstable when asked for a grid smaller than
the visible board, which caused the earlier intermittent failures.)

Usage:
  python evaluate_depth.py --rgb rgb_0002.png --depth depth_0002.png \
      --truth-mm 325.7 [--debug]
"""
import argparse
import os
import cv2
import numpy as np

DEPTH_SCALE_MM = 1.0        # D435 default: 100 um per raw unit


def a_star_channel(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32)
    ch = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.GaussianBlur(ch, (5, 5), 0)


def find_largest_grid(ch, max_pat=(30, 26), min_pat=(20, 16)):
    """Sweep pattern sizes, largest area first; return first detection."""
    sizes = [(pw, ph)
             for pw in range(max_pat[0], min_pat[0] - 1, -1)
             for ph in range(max_pat[1], min_pat[1] - 1, -1)]
    sizes.sort(key=lambda s: s[0] * s[1], reverse=True)
    for pat in sizes:
        ok, corners = cv2.findChessboardCornersSB(ch, pat)
        if ok:
            return corners, pat
    return None, None


def board_mask(img_bgr, debug=False, shrink_px=6):
    ch = a_star_channel(img_bgr)
    corners, pat = find_largest_grid(ch)
    if corners is None:
        if debug:
            os.makedirs("debug", exist_ok=True)
            cv2.imwrite(os.path.join("debug", "a_star.png"), ch)
        raise RuntimeError(
            "No checkerboard grid found. Run with --debug and inspect "
            "debug/a_star.png - the board must be visible and in focus.")
    print(f"Board detected: grid {pat[0]}x{pat[1]} inner corners")
    if debug:
        os.makedirs("debug", exist_ok=True)
        vis = img_bgr.copy()
        cv2.drawChessboardCorners(vis, pat, corners, True)
        cv2.imwrite(os.path.join("debug", "detected_overlay.png"), vis)

    hull = cv2.convexHull(corners.reshape(-1, 2).astype(np.int32))
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    k = 2 * shrink_px + 1
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--truth-mm", type=float, required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.rgb)
    if img is None:
        raise SystemExit(f"Could not read {args.rgb}")
    mask = board_mask(img, debug=args.debug) > 0

    raw = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise SystemExit(f"Could not read {args.depth}")
    if raw.dtype != np.uint16:
        print(f"WARNING: depth image is {raw.dtype}, expected uint16 (z16)")
    if raw.shape[:2] != img.shape[:2]:
        print(f"WARNING: depth {raw.shape[:2]} vs RGB {img.shape[:2]} size "
              "mismatch - mask may not align (use aligned streams)")

    depth_mm = raw.astype(np.float64) * DEPTH_SCALE_MM
    z = depth_mm[mask]
    z = z[z > 0]

    e = z - args.truth_mm
    rms = np.sqrt(np.mean(e ** 2))

    print(f"Pixels used:     {len(e)} / {int(mask.sum())} "
          f"(fill {100.0 * len(e) / mask.sum():.2f} %)")
    print(f"RMS depth error: {rms:.3f} mm")
    print(f"  bias (mean):   {e.mean():+.3f} mm")
    print(f"  noise (std):   {e.std():.3f} mm")
    print(f"  max |error|:   {np.abs(e).max():.3f} mm")
    print(f"  median depth:  {np.median(z):.1f} mm")


if __name__ == "__main__":
    main()