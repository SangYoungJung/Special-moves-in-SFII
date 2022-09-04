import cv2
import os
from datetime import datetime

input  = "./dataset/video/2022-07-14 21-05-17.mp4"
output = "./dataset/image/"
stride = 15 # every number / 30 second

vidcap = cv2.VideoCapture(input)
print('Frame width:', int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print('Frame height:', int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('Frame count:', int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)))
print('FPS:',int(vidcap.get(cv2.CAP_PROP_FPS)))

success, image = vidcap.read()
count = 0
while success:
    if count % int(stride) == 0:
        name = '{}{}_{}_{:08}.jpg'.format(output, 
                            os.path.basename(input), 
                            str(int(datetime.now().timestamp())),
                            count)
        ret = cv2.imwrite(name, image)
        if count % 100 == 0: print(ret,name)
        if ret != True: break
    success,image = vidcap.read()
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'): break

vidcap.release()
cv2.destroyAllWindows()