import os
import cv2

def extract_keyframes(video_path, min_frame_difference=15000000):
    vid = cv2.VideoCapture(video_path)
    keyframes = []  # Initialize a list to store keyframes

    count = 0
    flag = 0
    x = 7  # Save every x frames

    prev_frame = None

    while True:
        exist, image = vid.read()

        if not exist:
            break

        if flag % x == 0:
            keyframes.append(image)  # Add the frame to the list of keyframes

        if prev_frame is not None:
            frame_difference = cv2.absdiff(prev_frame, image).sum()

            if frame_difference > min_frame_difference:
                keyframes.append(image)  # Add the frame to the list of keyframes

        flag += 1
        prev_frame = image

    vid.release()
    return keyframes


def display_keyframes(keyframes):
    for i, keyframe in enumerate(keyframes):
        cv2.imshow(f"Keyframe {i + 1}", keyframe)
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()

def main():
    video_path = "D:/git/video_to_frame_augmentation/Kathakali dataset video/Adbutham/Adbutham 1.mp4"  # Replace with the actual video path
    keyframes = extract_keyframes(video_path)

    if not keyframes:
        print("No keyframes extracted.")
    else:
        print(f"Extracted {len(keyframes)} keyframes from {video_path}.")
        display_keyframes(keyframes)

if __name__ == "__main__":
    main()




