import os
import cv2
def extract_keyframes(video_path, n=120):
    vid = cv2.VideoCapture(video_path)
    keyframes = []  # Initialize a list to store keyframes

    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # If the video has fewer frames than n, duplicate frames to achieve the desired count
    if total_frames <= n:
        print("Frames are less than ", n)
        while len(keyframes) < n:
            exist, frame = vid.read()
            if not exist:
                break
            keyframes.append(frame)

    else:
        # Extract exactly n keyframes if the video is larger than n
        print("Frames are greater than ", n)
        frame_indices = [int(i * total_frames / n) for i in range(n)]
        for frame_index in frame_indices:
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            exist, frame = vid.read()
            if not exist:
                break
            keyframes.append(frame)

    vid.release()
    return keyframes

def display_keyframes(keyframes, delay=50):
    if not keyframes:
        print("No keyframes to display.")
        return

    for frame in keyframes:
        cv2.imshow("Keyframe Montage", frame)
        cv2.waitKey(delay)  # Adjust the delay (in milliseconds) as needed

    cv2.destroyAllWindows()


def main():
    video_path = "D:/git/video_to_frame_augmentation/Kathakali dataset video/Shantham/Shantham 2.mp4"
    keyframes = extract_keyframes(video_path)

    if not keyframes:
        print("No keyframes extracted.")
    else:
        print(f"Extracted {len(keyframes)} keyframes from {video_path}.")
        display_keyframes(keyframes)

if __name__ == "__main__":
    main()
