import cv2
import mediapipe as mp

mp_holistic=mp.solutions.holistic
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles

def draw_styled_landmarks(image, results):
	mp_drawing.draw_landmarks(image,
							  results.pose_landmarks,
							  mp_holistic.POSE_CONNECTIONS
							  )

cap=cv2.VideoCapture(0)
with mp_holistic.Holistic(
	static_image_mode=False,
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5) as holistic:
	while cap.isOpened():
		success, image=cap.read()

		if not success:
			print("Ignoring Empty Camera!")
			continue
		image.flags.writeable=False
		image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		results=holistic.process(image)

		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		mp_drawing.draw_landmarks(
			image,
			results.pose_landmarks,
			mp_holistic,
		draw_styled_landmarks(image, results))
		mp_drawing.draw_landmarks(
			image,
			results.pose_landmarks,
			mp_holistic,
		draw_styled_landmarks(image, results))
		cv2.imshow(image, results)
		if cv2.waitKey(10) & 0xFF==ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()