
import cv2 as cv
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PlotModule import LivePlot



detector = FaceMeshDetector()    # more advance detection technique
# detector = FaceDetector()
plot = LivePlot(640,480,[20,50])
plot_right = LivePlot(640,480,[20,50])


cam=cv.VideoCapture("test3.mp4")
# cam=cv.VideoCapture(0)

eye_id = [22,23,24,26,110,157,158,159,160,161,130,243 , 466 , 414 ,398 ,390,388,387 ,385,384,382,381,380,374 ,263]

ratio_list_left = []  # smothening plot
ratio_list_right = []  # smothening plot
blink_counter = 0
blink_counter_pause_frame = 0

color_red= (0,0,255)
color_green = (0,255,0)


while True:
    ist , f =  cam.read() 
    real_frame = f.copy()
    real_frame_left = f.copy()
    real_frame_right = f.copy()
    

    
    # f , face = detector.findFaces(f)
    f , face = detector.findFaceMesh(f)
    
    
    #  if face gound in frame
    if face:
        faces = face[0]      # consider only 1 face
        
        # left eye coordinates    
        left_eye_up = faces[159]
        left_eye_down = faces[23]
        left_eye_left = faces[130]
        left_eye_right = faces[243]
        left_all = [left_eye_left,left_eye_up,left_eye_down,left_eye_right]
        
  
           
        # right_eye coordinates
        right_eye_up = faces[386]
        right_eye_down = faces[374]
        right_eye_left = faces[398]
        right_eye_right = faces[263]
        right_all = [right_eye_left,right_eye_up,right_eye_down,right_eye_right]
        
        #   #for drawing circles around eyes
        for e in left_all:
            cv.circle(real_frame,e,5,(0,0,255),cv.FILLED,1)
        
        for e in right_all:
            cv.circle(real_frame,e,5,(0,0,255),cv.FILLED,1)
        
        #  Draw lines betn eye's up down and left right
        cv.line(real_frame_left,left_eye_up,left_eye_down,(0,200,0),1)
        cv.line(real_frame_left,left_eye_left,left_eye_right,(0,200,0),1)
        cv.line(real_frame_right,right_eye_up,right_eye_down,(0,200,0),1)
        cv.line(real_frame_right,right_eye_left,right_eye_right,(0,200,0),1)
        
        #  find distance between eye's up and down
        length_vertical , _ = detector.findDistance(left_eye_up,left_eye_down)
        length_horizonatl , _ = detector.findDistance(left_eye_left,left_eye_right)
        
        length_vertical_right , _ = detector.findDistance(right_eye_up,right_eye_down)
        length_horizonatl_right , _ = detector.findDistance(right_eye_left,right_eye_right)
        
        ratio_left = (length_vertical/length_horizonatl)*100
        
        ratio_right = (length_vertical_right/length_horizonatl_right)*100
        
        ratio_list_left.append(ratio_left)
        
        if len(ratio_list_left) > 3:
            ratio_list_left.pop(0)
                
        # print("ratio_left: ",ratio_left)
        actual_left_ratio =sum(ratio_list_left)/len(ratio_list_left)
      
        
        img_plot = plot.update(actual_left_ratio)
        img_plot_right = plot_right.update(ratio_right)
        
        
        if actual_left_ratio>30:
            cv.putText(img_plot , f"For Left Eye" , (100,100),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            cv.putText(img_plot_right , f"For Right Eye" , (100,100),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        else:
            cv.putText(img_plot , f"For Left Eye" , (100,100),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            cv.putText(img_plot_right , f"For Right Eye" , (100,100),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            if blink_counter_pause_frame ==0:
                blink_counter += 1
                blink_counter_pause_frame= 1
                cv.line(real_frame_left,left_eye_up,left_eye_down,color_red,1)
                cv.line(real_frame_left,left_eye_left,left_eye_right,color_red,1)
                # time.sleep(0.1)
        
        if blink_counter_pause_frame != 0:
            blink_counter_pause_frame +=1
            if blink_counter_pause_frame > 15:
                blink_counter_pause_frame = 0
        
        
        
        #  cropped part where EYE is located , imgcrop = [y:y+h , x:x+w]
        y= left_eye_up[1]-10
        h =left_eye_down[1] - left_eye_up[1] +20
        w = left_eye_right[0] - left_eye_left[0] +20
        x=  left_eye_left[0]-10
        real_frame_left = real_frame_left[ y:y+h , x:x+w]
        
        
        y= right_eye_up[1]-10
        h =right_eye_down[1] - right_eye_up[1] +20
        w = right_eye_right[0] - right_eye_left[0] +20
        x=  right_eye_left[0]-10
        real_frame_right =  real_frame_right[ y:y+h , x:x+w]
        
        real_frame_left = cv.resize(real_frame_left,(400,400))
        cv.putText(real_frame_left, "Left EYE", (10,50), cv.FONT_HERSHEY_SIMPLEX, 1, (200,255,255), 1)
        real_frame_right = cv.resize(real_frame_right,(400,400))
        cv.putText(real_frame_right, "Right EYE", (10,50), cv.FONT_HERSHEY_SIMPLEX, 1, (200,255,255), 1)
        
        
        real_frame=cv.resize(real_frame,(400,400))
        eyes_combined = cv.hconcat([real_frame_left,real_frame,real_frame_right])
        
        img_plot = cv.resize(img_plot,(600,300))
        img_plot_right = cv.resize(img_plot_right,(600,300))
        img_plot_total = cv.hconcat([img_plot,img_plot_right])
        
        eyeAndpolt_combined = cv.vconcat([img_plot_total,eyes_combined])

        
        frame_shown = eyeAndpolt_combined    # 1200, 700
        # cvzone.putTextRect(frame_shown , f"Blink Counter: {blink_counter}" , (420,350) ,1 ,2 )
      
        
    else:
        f= cv.resize(f,(1200, 700))
        frame_shown = f
        
    
        
    # # cv.putText(f, "FPS: "+str(int(cam.get(cv.CAP_PROP_FPS))), (10,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    # cv.putText(f, f"{left_eye_up} ", (10,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
    # cv.putText(f, f"{left_eye_down} ", (10,80), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
    # cv.putText(f, f"{length} ", (10,110), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)

    
    # cv.putText(f, "FRAME: "+str(int(cam.get(cv.CAP_PROP_FRAME_COUNT))), (20,80), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    # time.sleep(0.1)

    cv.imshow('Image',frame_shown )
    if cv.waitKey(20) & 0xFF== ord('d'):
        break

cv.waitKey(0)  