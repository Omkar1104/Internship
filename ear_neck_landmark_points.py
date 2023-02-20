import cv2
import mediapipe as mp

mp_holistic=mp.solutions.holistic
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def mediapipe_detection(image, model):
	image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image.flags.writeable=False
	results=model.process(image)
	image.flags.writeable=True
	image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	return image, results

def draw_landmarks(image, results):
	mp_drawing.draw_landmarks(
		image, results.pose_landmarks, mp_holistic)
	mp_drawing.draw_landmarks(
		image, results.pose_landmarks, mp_holistic)


def draw_styled_landmarks(image, results):
	mp_drawing.draw_landmarks(
		image,
		results.pose_landmarks,
		mp_holistic.POSE_CONNECTIONS,
		mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=2))

cap=cv2.VideoCapture(0)

with mp_holistic.Holistic(
		static_image_mode=False,
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5) as holistic:
	while cap.isOpened():
		ret, frame=cap.read()
		image, results=mediapipe_detection(frame, holistic)

		left_ear_x=results.pose_landmarks.landmark[7].x
		left_ear_y=results.pose_landmarks.landmark[7].y
		right_ear_x = results.pose_landmarks.landmark[8].x
		right_ear_y = results.pose_landmarks.landmark[8].y
		left_shoulder_x=results.pose_landmarks.landmark[11].x
		left_shoulder_y=results.pose_landmarks.landmark[11].y
		left_shoulder_z=results.pose_landmarks.landmark[11].z
		right_shoulder_x=results.pose_landmarks.landmark[12].x
		right_shoulder_y=results.pose_landmarks.landmark[12].y
		right_shoulder_z=results.pose_landmarks.landmark[12].z
		shoulder_mid_x=(left_shoulder_x+right_shoulder_x)/2
		shoulder_mid_y=(left_shoulder_y+right_shoulder_y)/2
		shoulder_mid_z=(left_shoulder_z+right_shoulder_z)/2
		chin_x=results.face_landmarks.landmark[152].x
		chin_y=results.face_landmarks.landmark[152].y
		chin_z=results.face_landmarks.landmark[152].z
		neck_mid_x=(shoulder_mid_x+chin_x)/2
		neck_mid_y=(shoulder_mid_y+chin_y)/2
		neck_mid_z=(shoulder_mid_z+chin_z)/2

		image_height, image_width, _ = image.shape
		image_rightear_x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].x * image_width-20
		image_rightear_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].y * image_height+27
		image_leftear_x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x * image_width+12
		image_leftear_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y * image_height+27
		image_neck_x = neck_mid_x * image_width
		image_neck_y = neck_mid_y * image_height

		cv2.circle(image, (int(image_neck_x), int(image_neck_y)), 2, [0, 0, 255], 2)
		cv2.circle(image, (int(image_leftear_x), int(image_leftear_y)), 2, [0, 0, 255], 2)
		cv2.circle(image, (int(image_rightear_x), int(image_rightear_y)), 2, [0, 0, 255], 2)

		if results.pose_landmarks:
			print(
				f'Neck Midpoint coordinates ('
				f'{image_neck_x}, '
				f'{image_neck_y})'
			)
			print(
				f'Right Ear coordinates ('
				f'{image_rightear_x}, '
				f'{image_rightear_y})'
			)
			print(
				f'Left Ear coordinates ('
				f'{image_leftear_x}, '
				f'{image_leftear_y})'
			)
			print()
		draw_styled_landmarks(image, results)

		cv2.imshow('OpenCV Feed', image)

		if cv2.waitKey(10) & 0xFF==ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()






