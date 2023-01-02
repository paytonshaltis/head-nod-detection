# Payton Shaltis
# Head Movement Detection
# ---
# Determines if the user is nodding (YES) or shaking their head (NO) and 
# prints a message to the console, roughly one message per shake or nod.
# Based on the supplied sample. New code is indicated with comments.

from operator import attrgetter
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
frame = 0

# --- NEWLY ADDED CODE START --- #

# Constants for tweaking program.
FRAMES_TO_ANALYZE = 10
NODDING_SENSITIVITY = 0.0125
SHAKING_SENSITIVITY = 0.02

VERTICAL_ADJUSTMENT = 0.2
HORIZONTAL_ADJUSTMENT = 0.12

# Lists to store frame data.
nodding_coordinates = []
shaking_coordinates = []

"""
Returns the number of times data[coord] changes directions (increasing,
decreasing) when read sequentially. The parameter 'sensitivity'
prevents insignificantly small changes in direction from counting.
"""
def direction_changes(data, coord, sensitivity):
  current_data = None
  prev_data = None
  current_direction = None
  prev_direction = None
  peak_or_valley = getattr(data[0], coord)
  num_direction_changes = 0

  # Traverse the entire list of data.
  for i in range(len(data)):

    current_data = getattr(data[i], coord)
    if prev_data:

      # If the two neighboring data points are significantly far away...
      if(abs(peak_or_valley - current_data) > sensitivity):

        # Determine the direction of travel (variables won't be equivalent).
        if(peak_or_valley > current_data):
          current_direction = 'increasing'
        else:
          current_direction = 'decreasing'

        # Determine if there has been a direcitonal change.
        if(prev_direction and current_direction != prev_direction):

          # Assign a new peak or valley as the current data point.
          num_direction_changes += 1
          peak_or_valley = current_data
        
        # For the first time a direction is established.
        elif(not prev_direction):
          prev_direction = current_direction
          peak_or_valley = current_data

    prev_data = current_data

  # Return the total number of significant directional changes.
  return num_direction_changes

# --- NEWLY ADDED CODE END --- #

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())

        # --- NEWLY ADDED CODE START --- #

        # Keep track of important landmarks.
        chin = face_landmarks.landmark[199]
        sidehead = face_landmarks.landmark[447]

        # Used to adjust the sensitivity constants based on distance to screen.
        tophead = face_landmarks.landmark[10]
        bottomhead = face_landmarks.landmark[152]
        distance_adjustment = (bottomhead.y - tophead.y) / 0.5

        # Always append a new frame to both lists.
        nodding_coordinates.append(chin)
        shaking_coordinates.append(sidehead)

        # Ready to analyze frames.
        if(len(nodding_coordinates) > FRAMES_TO_ANALYZE and len(shaking_coordinates) > FRAMES_TO_ANALYZE):

          # Pop the oldest frame from both lists (just looking at the last FRAMES_TO_ANALYZE frames).
          nodding_coordinates.pop(0)
          shaking_coordinates.pop(0)

          # Head nod has occurred:
          # Must have (1) at least one nodding direction change, (2) no shaking direction changes, 
          # and (3) chin must not move up the y axis significantly; reduces up / down motion detection.
          if(direction_changes(nodding_coordinates, "z", NODDING_SENSITIVITY * distance_adjustment) > 0 
              and direction_changes(shaking_coordinates, "z", SHAKING_SENSITIVITY * distance_adjustment) == 0
              and abs(max(nodding_coordinates, key=attrgetter('y')).y - min(nodding_coordinates, key=attrgetter('y')).y) 
                <= VERTICAL_ADJUSTMENT * distance_adjustment):
            print("YES")
            nodding_coordinates = []
            shaking_coordinates = []

          # Head shake has occurred:
          # Must have (1) at least one shaking direction change, (2) no nodding direction changes, 
          # and (3) side of head must not move across the X axis significantly; reduces left / right motion detection.
          elif(direction_changes(shaking_coordinates, "z", SHAKING_SENSITIVITY * distance_adjustment) > 0 
              and direction_changes(nodding_coordinates, "z", NODDING_SENSITIVITY * distance_adjustment) == 0
              and abs(max(shaking_coordinates, key=attrgetter('x')).x - min(shaking_coordinates, key=attrgetter('x')).x) 
                <= HORIZONTAL_ADJUSTMENT * distance_adjustment):
            print("NO")
            nodding_coordinates = []
            shaking_coordinates = []
        
        # --- NEWLY ADDED CODE END --- #

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

