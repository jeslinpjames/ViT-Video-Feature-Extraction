import os

def extract_keyframes(video_path, output_dir, min_frame_difference=19000000):
    vid = cv2.VideoCapture(video_path)
    count = 0
    flag = 0
    x = 7  # Save every x frames

    prev_frame = None

    while True:
        exist, image = vid.read()

        if not exist:
            break

        if flag % x == 0:
            label = os.path.join(output_dir, f"frame_{count}.jpeg")
            cv2.imwrite(label, image)
            count += 1

        if prev_frame is not None:
            frame_difference = cv2.absdiff(prev_frame, image).sum()

            if frame_difference > min_frame_difference:
                keyframe_label = os.path.join(output_dir, f"keyframe_{count}.jpeg")
                cv2.imwrite(keyframe_label, image)
                count += 1

        flag += 1
        prev_frame = image

    vid.release()
    print(f"Keyframes extracted from {video_path}.")