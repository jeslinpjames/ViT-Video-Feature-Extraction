import os
import cv2

def extract_keyframes(video_path, n=120):
    vid = cv2.VideoCapture(video_path)
    keyframes = []  # Initialize a list to store keyframes

    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # If the video has fewer frames than n, duplicate the last frame continuously
    if total_frames <= n:
        print("Frames are less than ",n)
        while len(keyframes) < n:
            exist, frame = vid.read()
            if not exist:
                break
            keyframes.append(frame)

        # Duplicate the last frame continuously until reaching n keyframes
        while len(keyframes) < n:
            keyframes.append(keyframes[-1])

    else:
        # Adjust min_frame_difference to extract exactly n keyframes if the video is larger than n
        min_frame_difference = total_frames // n
        flag = 0
        prev_frame = None
        print("Frames are greater than ",n)

        while len(keyframes) < n:
            exist, image = vid.read()

            if not exist:
                break

            # Ensure both frames have the same shape before calculating the absolute difference
            if prev_frame is not None and prev_frame.shape == image.shape:
                frame_difference = cv2.absdiff(prev_frame, image).sum()

                if flag == 0 or frame_difference > min_frame_difference:
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




