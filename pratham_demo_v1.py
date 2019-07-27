import numpy as np
import cv2 
import os
import flask
import ast
from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
photos = UploadSet('photos', IMAGES)
app = flask.Flask(__name__)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/'
configure_uploads(app, photos)
app.config['JSON_SORT_KEYS'] = False

def load_digit_model():
    json_file = open('model/digit_recognition.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('model/digit_recognition.h5')
    return loaded_model

def generate_metadata():
    metadata={}
    metadata["page_1"] = [{"x":184 , "y":414 , "h":29 , "w":85, "qno":"1.1", "type":"digit_rec"},
                        {"x":400 , "y":414 , "h":29 , "w":85, "qno":"1.2", "type":"digit_rec"},
                        {"x":183 , "y":624 , "h":28 , "w":85, "qno":"2.1", "type":"digit_rec"},
                        {"x":399 , "y":624 , "h":28 , "w":86, "qno":"2.2", "type":"digit_rec"}]
    metadata["page_2"]=  [{"x":213 , "y":336 , "h":29 , "w":86, "qno":"3", "type":"digit_rec"},
                        {"x":212 , "y":625 , "h":28 , "w":86, "qno":"4", "type":"digit_rec"}]
    metadata["page_3"] = [{"x":236 , "y":219 , "h":29 , "w":85, "qno":"5.1", "type":"digit_rec"},
                        {"x":235 , "y":332 , "h":30 , "w":86, "qno":"5.2", "type":"digit_rec"},
                        {"x":233 , "y":502 , "h":30 , "w":86, "qno":"6.1", "type":"digit_rec"},
                        {"x":232 , "y":624 , "h":29 , "w":86, "qno":"6.2", "type":"digit_rec"}]
    metadata["page_4"]=  [{"x":353 , "y":315 , "h":45 , "w":32, "qno":"7", "type":"digit_rec_ver"},
                        {"x":353 , "y":450 ,"h": 45 , "w":32, "qno":"8.1", "type":"digit_rec_ver"},
                        {"x":113 , "y":580 , "h":47 , "w":76, "qno":"8.2", "type":"shaded_frac","rows":1,"cols":3,"parts":2,"frac_type":"square"}]
    metadata["page_5"] = [{"x":270 , "y":298 , "h":25 , "w":125, "qno":"9", "type":"OMR", "xrange":{'A':42, 'B': 72, 'C': 102, 'D': 132 }},
                        {"x":180 , "y":408 , "h":210, "w":100, "qno":"10.1", "type":"matching_pair"}]
    metadata["page_6"] = [{"x":307 , "y":391 , "h":29 , "w":65, "qno":"10.2 ", "type":"digit_rec"}]

    return metadata

def show_image(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rectify(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1) 
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

def four_point_transform(image, target):
    # obtain a consistent order of the points and unpack them
    # individually
    approx = rectify(target)
    (tl, tr, br, bl) = approx

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    pts1 = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")


    M = cv2.getPerspectiveTransform(approx,pts1)

    warped = cv2.warpPerspective(image,M,(maxWidth, maxHeight))
    # return the warped image
    return warped,M

def sort_contours(contours_hierarchy,imagecpy):
    square_contours=[]
    contours_hierarchy = list(contours_hierarchy)
    for c in contours_hierarchy:
        perimeter = cv2.arcLength(c[0], True)
        approx = cv2.approxPolyDP(c[0], 0.02*perimeter, True)
        if len(approx)!=4:
            square_contours.append(c)
    return square_contours

def get_warped_image(image):
    RESCALED_HEIGHT = 800.0
    ratio = image.shape[0] / RESCALED_HEIGHT
    rescaled_image = imutils.resize(image, height=int(RESCALED_HEIGHT))
    points = detect_outerblobs(rescaled_image)
    warped,M = four_point_transform(rescaled_image, np.array(points))
    return warped,M

def box_extraction(image,page_type,start_point,end_point):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.fastNlMeansDenoising(gray,None,7,21)

    _,thresh = cv2.threshold(denoised_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    imagecpy = image.copy()

    cnt_hierarchy = zip(contours,hierarchy[0])
    
    cnt_hierarchy = sort_contours(cnt_hierarchy,imagecpy)
    print(hierarchy[0])

    l = []
    roilist = []
    letter_contours = []
    for index,hierarchy in enumerate(cnt_hierarchy):
        h = hierarchy[1]
#(h[0] > -1 and h[1]) > -1 or (h[0]>-1 and h[2]>-1)
        if (h[0]>-1 and h[3]<0):
            letter_contours.append(contours[index])
        elif h[0]==-1:
            outer_contour = contours[index]
            
    if page_type== "digit_rec":
        letter_contours = sorted(letter_contours,key = lambda lc:cv2.boundingRect(lc)[0])
        position = ["None"]*len(letter_contours)
        contour_positions = []
        for index,lc in enumerate(letter_contours):
            contour_positions.append({"contour":lc,"position":position[index]})
        
    elif page_type == "digit_rec_ver":
        contour_positions = []

        letter_contours = sorted(letter_contours,key = lambda lc:cv2.boundingRect(lc)[1])
        if len(letter_contours)==1:
            letter_contours = np.vstack(letter_contours).squeeze()
            xo,yo,wo,ho = cv2.boundingRect(outer_contour)
            xi,yi,wi,hi = cv2.boundingRect(letter_contours)
            
            num_dsit =abs(yi-yo)
            
            deno_dist = abs(yi-(yo+ho))
            print("num {}, den {}".format(num_dsit,deno_dist))
            position = "numerator" if num_dsit<deno_dist else  "denominator"
            contour_positions = [{"contour":letter_contours,"position":position}]
        else:
            position = [None]*len(letter_contours)
            position[0] = "numerator"
            position[1] = "denominator"
            for lc in letter_contours:
                contour_positions.append({"contour":lc,"position":position})
       
        
    for item in contour_positions:
        x,y,w,h = cv2.boundingRect(item["contour"])
        show_image("imagecpy_box",thresh[y-1:y+h+1,x-1:x+w+1])
        roilist.append({"image":thresh[y-1:y+h+1,x-1:x+w+1],"position":item["position"]})
    
    return roilist

def get_square_frac(square_fraction,num_rows,num_cols):
    square_fraction_gray = cv2.cvtColor(square_fraction,cv2.COLOR_BGR2GRAY)
    square_fraction_denoised_gray = cv2.fastNlMeansDenoising(square_fraction_gray, None, 7, 21)
    ret,thresh_square = cv2.threshold(square_fraction_denoised_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)

    hist_sums = []
    height,width = thresh_square.shape

    desired_height = int(height/num_rows)

    desired_width = int(width/num_cols)

    for x in range(0,width-desired_width+1,desired_width):
        for y in range(0,height-desired_height+1,desired_height):
            if x>0:
                rect = thresh_square[y:y+desired_height-1,x-1:x+desired_width-1]
            else:
                rect = thresh_square[y:y+desired_height-1,x:x+desired_width-1]

            hist_row = np.count_nonzero(255-rect,axis=0)
            hist_col = np.count_nonzero(255-rect,axis=1)
            hist_sums.append(sum(hist_row)+sum(hist_col))
    shaded_indices = [x for x in hist_sums if x>500 and x>=np.max(hist_sums)*0.5]
    return str(len(shaded_indices))+"/"+str(num_rows*num_cols)

def get_shaded_regions(circle_fraction,center,radius,parts,angle):
    # need to know how many parts are asked 
    hist_sums = []
    images_list=[]
    additon_angle = 360/parts
    for i in range(int(angle),-360,-int(additon_angle)):
        deg1=i
        deg2 = deg1-additon_angle
        color=(255,255,255)
        axes= (radius,radius)

        mask = np.zeros(circle_fraction.shape,np.uint8)
        circle_mask = np.zeros(circle_fraction.shape,np.uint8)
        cv2.ellipse(circle_mask,center, axes,0, deg1, deg2, color,-1,8,0);

        mask_res = cv2.bitwise_and(circle_fraction,circle_mask)


        mask_res[np.where(mask_res==0)]=255


        new_image_gray = cv2.cvtColor(mask_res,cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit= 3.0, tileGridSize =(8,8))
        roi_hist_eq = clahe.apply(new_image_gray)

        ret,new_image_thresh = cv2.threshold(roi_hist_eq,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


        images_list.append(new_image_thresh)
        hist_row = np.count_nonzero(255 - new_image_thresh,axis=0)
        hist_col = np.count_nonzero(255 - new_image_thresh,axis=1)
        hist_sums.append(np.sum(hist_row)+np.sum(hist_col))
    shaded_indices = [x for x in hist_sums if x>1500 and x>np.max(hist_sums)*0.8]
    return str(len(shaded_indices))+"/"+str(parts)
def get_circle_frac(circle_fraction,parts,init_angle):
    circle_frac_gray = cv2.cvtColor(circle_fraction,cv2.COLOR_BGR2GRAY)
    circle_frac_denoised_gray = cv2.fastNlMeansDenoising(circle_frac_gray, None, 7, 21)
    ret,circle_frac_thresh = cv2.threshold(circle_frac_denoised_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    canny = cv2.Canny(circle_frac_gray,0,50)
    circles=[]
    (_,contours,_) = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours,key=cv2.contourArea,reverse=True)
    for contour in contours:
        if(len(contour)>4):
            ellipse = cv2.fitEllipse(contour)
            (x,y),(MA,ma),angle  = cv2.fitEllipse(contour)
                #add it
            a = ma/2
            b = MA/2

            circle_fractioncpy = circle_fraction.copy()

            eccentricity = sqrt(pow(a,2)-pow(b,2))
            eccentricity = round(eccentricity/a,2)



            if(eccentricity <= 0.5):
                circles.append(contour)


    circle_fractioncpy = circle_fraction.copy()

    circle_outer = sorted(circles,key=cv2.contourArea,reverse=True)[0]


    # # cropping using circle

    (x,y),radius = cv2.minEnclosingCircle(circle_outer)
    center = (int(x),int(y))
    radius = int(radius)
    x = int((x - radius))
    y = int((y - radius))
    return get_shaded_regions(circle_fraction,center,radius,parts,init_angle)

def get_omr_response(image, omr_range):
    
    omr_range = ast.literal_eval(omr_range)

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10;
    params.maxThreshold = 200;
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 400
    params.filterByCircularity = True
    params.maxCircularity = 0.9
    params.filterByConvexity = True
    params.minConvexity = 0.87
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(image)
    #im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    detected_point = keypoints[0].pt
    #print(detected_point)
    if (len(keypoints)) == 0:
        return ""
    else :
        for key,val in omr_range.items():
            if(detected_point[0] < val):
                return key
def detect_outerblobs(image):
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 200;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 200
    params.maxArea = 400

    # Filter by Circularity
    params.filterByCircularity = True
    params.maxCircularity = 0.9

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(image)
    objects = []
    for k in keypoints:
        #print(k.pt)
        objects.append((int(k.pt[0]), int(k.pt[1])))

    points = np.array(objects)
    #print(points)

    max_x = max([_[0] for _ in points])
    min_x = min([_[0] for _ in points])

    max_y = max([_[1] for _ in points])
    min_y = min([_[1] for _ in points])

    points = [[min_x, min_y],
                [max_x, min_y],
                 [max_x, max_y],
                 [min_x, max_y]
             ]
    return points

def get_warped_image(image):
    RESCALED_HEIGHT = 800.0
    ratio = image.shape[0] / RESCALED_HEIGHT
    rescaled_image = imutils.resize(image, height=int(RESCALED_HEIGHT))
    points = detect_outerblobs(rescaled_image)
    warped,M = four_point_transform(rescaled_image, np.array(points))
    return warped,M
def get_matching_pairs(image):
    image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_copy=image.copy()
    thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                    cv2.THRESH_BINARY_INV,11,2)
    lines = cv2.HoughLinesP(thresh,1,np.pi/180,threshold=30,minLineLength=30,maxLineGap=30)
    t1=[]
    init_points = []
    final_points = []
    points=[]
    
    if not np.any(lines):
        return("No matches")
    else:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            points.append([ x1,y1,x2,y2])
    points = sorted(points,key=lambda x:x[1])
    print(points)
    init_y,end_y=points[0][1],points[0][1]

    duplicate_points=[]
    for i in range(0,len(points)-1):
        for j in range(i+1,len(points)):
            if points[j][1] in range(points[i][1],points[i][1]+30):
                 points[j][1] = points[i][1]
    print("points",points)
    duplicate_distances = [{"x1":x1,"y1":y1,"x2":x2,"y2":y2,"distance":sqrt((y2-y2)**2+(x2-x1)**2)} for x1,y1,x2,y2 in points]
    print("duplicate_distances",duplicate_distances)
    unique_y1={item['y1'] for item in duplicate_distances}
    final_points = []
    for y1 in unique_y1:
        items = [item for item in duplicate_distances if item["y1"]==y1]
        index = np.argmax([item["distance"] for item in items])
        init_points.append(items[index]["y1"])
        final_points.append(items[index]["y2"])
    temp_order=[i[0] for i in sorted(enumerate(init_points),key=lambda x:x[1])]
    init_points_sorted_indices=sorted(range(len(temp_order)),key=lambda x:temp_order[x])
    init_points = list(zip(init_points,init_points_sorted_indices))
    print(init_points)

    temp_order=[i[0] for i in sorted(enumerate(final_points),key=lambda x:x[1])]
    final_points_sorted_indices=sorted(range(len(temp_order)),key=lambda x:temp_order[x])
    final_points = list(zip(final_points,final_points_sorted_indices))
    print(final_points)
    match_string = ""
    for index in range(len(final_points)):
        print(init_points[index][1],"->",final_points[index][1])
        match_string+=str(init_points[index][1])+"->"+str(final_points[index][1])
        if index< (len(final_points)-1):
            match_string+=","
    return match_string

@app.route("/get_response", methods=["POST"])
def get_response():
    data = {"success": False}
    if flask.request.method == "POST" and "image" in request.files and "page_number" in request.values:
        filepath = photos.save(request.files["image"])
        print(request.values.get("page_number"))
        image = cv2.imread("static/"+filepath)
        warped,M = get_warped_image(image)
        show_image("warped",warped)
        metadata = generate_metadata()
        metadata_page = metadata["page_"+str(page)]
        margin = 3
        characters = ['0','1','2','3','4','5','6','7','8','9']
        model = load_digit_model()
        for metadata in metadata_page:
            x,y,w,h = metadata["x"],metadata["y"],metadata["w"],metadata["h"]
            qno = metadata["qno"]
            qtype = metadata["type"]
            rotated_point = M.dot(np.array((x,y) + (1,)))
            rotated_point = rotated_point.astype(int)
            cropped = warped[rotated_point[1]-margin:(rotated_point[1]+h)+margin, 
                             rotated_point[0]-margin:(rotated_point[0]+w)+margin]

            if "digit_rec" in qtype:
                new_point = rotated_point[:2]
                init_new_point = np.array(new_point).reshape(1,-1)
                end_new_point =  np.array([new_point[0]+w,new_point[1]+h]).reshape(1,-1)
                box_info = box_extraction(cropped,qtype,init_new_point,end_new_point)
                for info in box_info:
                    roi = cv2.resize(info["image"],dsize = (28,28), interpolation = cv2.INTER_AREA)
                    show_image("roi",roi)

                    roi = np.array(roi)
                    t = np.copy(roi)
                    t = t /255.0
                    t = 1 - t
                    t = t.reshape(1,784)
                    pred = model.predict_classes(t)
                    print(pred)
            elif qtype == "shaded_frac":
                if metadata["frac_type"] == "square":
                    shaded = get_square_frac(cropped,metadata["rows"],metadata["cols"])
                    print(shaded)
            elif qtype == "matching_pair":
                margin_init=15
                margin_end=10
                cropped = warped[rotated_point[1]+margin_init:(rotated_point[1]+h)-margin_init, 
                             rotated_point[0]+margin_end:(rotated_point[0]+w)-margin_end]
                show_image("cropped",cropped)

                matches = get_matching_pairs(cropped)
                print(matches)
            elif: que_type == "OMR":
                xrange =metadata["xrange"]
                ans = get_omr_response(fraction,xrange)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
        