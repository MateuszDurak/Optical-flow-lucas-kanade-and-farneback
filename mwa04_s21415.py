import argparse
import cv2
import numpy as np

def main(args):
    """
    :param args: path to video
    :return: void
    """
    video_path = args.input_video
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    video = cv2.VideoCapture(video_path)
    captured, frame1 = video.read()

    previous_frame_farneback = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    previous_frame_lucas_kanade = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(previous_frame_lucas_kanade, mask=None,
                                 **feature_params)
    color = np.random.randint(0, 255, (100, 3))
    hsv_f = np.zeros_like(frame1)
    hsv_lk = np.zeros_like(frame1)
    hsv_f[..., 1] = 255
    hsv_lk[..., 1] = 255
    while video.isOpened():
        captured, frame2 = video.read()
        # shows input video
        dim = set_window_size(25, frame2)
        cv2.imshow('Input frame',
                   cv2.resize(frame2, dim))
        # lucas-kanade method
        next_frame_lucas_kanade = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        good_new, rgb1 = lucas_kanade(frame2,
                                      color,
                                      hsv_lk,
                                      previous_frame_lucas_kanade,
                                      p0,
                                      next_frame_lucas_kanade)
        cv2.imshow('Lucas-Kanade', cv2.resize(rgb1, dim))
        previous_frame_lucas_kanade = next_frame_lucas_kanade.copy()
        p0 = good_new.reshape(-1, 1, 2)
        # farneback method
        next_frame_farneback = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        rgb = farneback(previous_frame_farneback,
                        next_frame_farneback,
                        hsv_f)
        cv2.imshow('Farneback',
                   cv2.resize(rgb, dim))
        previous_frame_farneback = next_frame_farneback
        # hotkeys
        import argparse        # q -> quit
        # s -> snapshot
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('opticalflow.png', frame2)

    captured.release()
    cv2.destroyAllWindows()


def lucas_kanade(frame2, color, hsv_lk, previous_frame_lucas_kanade, p0, next_frame_lucas_kanade):
    """
    :param frame2: consecutive frames
    :param color: color of lines
    :param hsv_lk: zeros array wtih frame shape
    :param previous_frame_lucas_kanade:
    :param p0: strongest corner in image
    :param next_frame_lucas_kanade: next frame
    :return: new fitting corners and image
    """
    # parameters to pass to LK algorythm
    lk_params = dict(winSize=(15, 15),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))
    p1, st, err = cv2.calcOpticalFlowPyrLK(previous_frame_lucas_kanade,
                                           next_frame_lucas_kanade,
                                           p0, None,
                                           **lk_params)
    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    # Draw lines
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        hsv_lk = cv2.line(hsv_lk, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame2 = cv2.circle(frame2, (int(a), int(b)), 5, color[i].tolist(), -1)
    rgb1 = cv2.cvtColor(hsv_lk, cv2.COLOR_HSV2BGR)
    return good_new, rgb1


def set_window_size(scale, frame):
    """
    :param scale: scale of output image
    :param frame: input frame to take shape of it
    :return: width and height
    """
    scale_percent = scale
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    return width, height


def farneback(previous_frame_farneback, next_frame_farneback, hsv_f):
    """
    :param previous_frame_farneback: previous frame
    :param next_frame_farneback: next frame
    :param hsv_f: zeros array with frame shape
    :return: image
    """
    flow_f = cv2.calcOpticalFlowFarneback(previous_frame_farneback,
                                          next_frame_farneback,
                                          None,
                                          0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow_f[..., 0], flow_f[..., 1])
    hsv_f[..., 0] = ang * 180 / np.pi / 2
    hsv_f[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv_f, cv2.COLOR_HSV2BGR)
    return rgb


def parse_arguments():
    """
    :return: argument(input video path)
    """
    parser = argparse.ArgumentParser(
        description='This script shows difference between Lucas-Kanade method and  Farnebacks algorithm'
                    ' To run script you need to provide the path to the video as -i argument.'
                    ' While script is running, use \'q\' to quit or \'s\' to snapshot')
    parser.add_argument('-i',
                        '--input_video',
                        type=str,
                        required=True,
                        help='Path to video')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())
