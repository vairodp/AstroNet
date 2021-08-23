import numpy as np
import tensorflow as tf
import cv2

from configs.train_config import loss_params, SCORE_THRESHOLD, MAX_NUM_BBOXES

def bbox_to_x1y1x2y2(bbox):
    # bbox = [x, y, w, h] --> bbox = [x1, y1, x2, y2]

    bbox_xy = bbox[..., 0:2]
    bbox_wh = bbox[..., 2:4]
    bbox_x1y1 = bbox_xy - bbox_wh / 2
    bbox_x2y2 = bbox_xy + bbox_wh / 2

    return tf.concat([bbox_x1y1, bbox_x2y2], axis=-1)

def bbox_to_xywh(bbox):
    # bbox = [x1, y1, x2, y2] --> bbox = [x, y, w, h]

    bbox_x1y1 = bbox[..., 0:2]
    bbox_x2y2 = bbox[..., 2:4]

    bbox_wh = bbox_x2y2 - bbox_x1y1
    bbox_xy = bbox_x1y1 + bbox_wh / 2

    return tf.concat([bbox_xy, bbox_wh], axis=-1)

def non_max_suppression(yolo_feats, yolo_max_boxes, yolo_iou_threshold, yolo_score_threshold):
    """
    Applies the non max suppression to YOLO features and returns predicted boxes
    Args:
        yolo_feats (List[Tuple[tf.Tensor]]): For each output stage, is a 3-tuple of 5D tensors corresponding to
            bbox (N,grid_x,grid_y,num_anchors,4),
            objectness (N,grid_x,grid_y,num_anchors,4),
            class_probs (N,grid_x,grid_y,num_anchors,num_classes),
        yolo_max_boxes (int): Maximum number of boxes predicted on each image (across all anchors/stages)
        yolo_iou_threshold (float between 0. and 1.): IOU threshold defining whether close boxes will be merged
            during non max regression.
        yolo_score_threshold (float between 0. and 1.): Boxes with score lower than this threshold will be filtered
            out during non max regression.
    Returns:
        List[tf.Tensor]: 4 Tensors(N,yolo_max_boxes) respectively describing boxes, scores, classes, valid_detections
    """
    bbox_per_stage, objectness_per_stage, class_probs_per_stage = [], [], []

    for stage_feats in yolo_feats:
        num_boxes = (
            stage_feats[0].shape[1] * stage_feats[0].shape[2] * stage_feats[0].shape[3]
        )  # num_anchors * grid_x * grid_y
        bbox_per_stage.append(
            tf.reshape(
                stage_feats[0],
                (tf.shape(stage_feats[0])[0], num_boxes, stage_feats[0].shape[-1]),
            )
        )  # [None,num_boxes,4]
        objectness_per_stage.append(
            tf.reshape(
                stage_feats[1],
                (tf.shape(stage_feats[1])[0], num_boxes, stage_feats[1].shape[-1]),
            )
        )  # [None,num_boxes,1]
        class_probs_per_stage.append(
            tf.reshape(
                stage_feats[2],
                (tf.shape(stage_feats[2])[0], num_boxes, stage_feats[2].shape[-1]),
            )
        )  # [None,num_boxes,num_classes]

    bbox = tf.concat(bbox_per_stage, axis=1)
    objectness = tf.concat(objectness_per_stage, axis=1)
    class_probs = tf.concat(class_probs_per_stage, axis=1)

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.expand_dims(bbox, axis=2),
        scores=objectness * class_probs,
        max_output_size_per_class=yolo_max_boxes,
        max_total_size=yolo_max_boxes,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold,
    )

    return [boxes, scores, classes, valid_detections]


'''
def old_nms(inputs, anchors, anchor_masks):
    iou_threshold = loss_params['iou_threshold']
    score_threshold = SCORE_THRESHOLD
    max_bbox_size = MAX_NUM_BBOXES

    output_small, output_medium, output_large = inputs
    output_small = decode_predictions(output_small, anchors[anchor_masks[0]])
    output_medium = decode_predictions(output_medium, anchors[anchor_masks[1]])
    output_large = decode_predictions(output_large, anchors[anchor_masks[2]])

    # flatten output to shape [batch_size, toto_grid_size, *]
    bbox_small, objectness_small, class_probs_small = flatten_output(output_small)
    bbox_medium, objectness_medium, class_probs_medium = flatten_output(output_medium)
    bbox_large, objectness_large, class_probs_large = flatten_output(output_large)

    # concat all output
    bbox = tf.concat([bbox_small, bbox_medium, bbox_large], axis=1)
    confidence = tf.concat([objectness_small, objectness_medium, objectness_large], axis=1)
    class_probs = tf.concat([class_probs_small, class_probs_medium, class_probs_large], axis=1)

    scores = confidence * class_probs

    bboxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=max_bbox_size,
        max_total_size=max_bbox_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )

    return bboxes, scores, classes, valid_detections
'''

def decode_predictions(pred, anchors):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, -1), axis=-1)

    box_xy = loss_params["sensitivity_factor"] * tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    box = tf.concat([box_xy, box_wh], axis=-1)

    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, box


def flatten_output(outputs):
    bbox, objectness, class_probs, _ = outputs
    bbox = tf.reshape(bbox, (tf.shape(bbox)[0], -1, tf.shape(bbox)[-1]))
    objectness = tf.reshape(objectness, (tf.shape(objectness)[0], -1, tf.shape(objectness)[-1]))
    class_probs = tf.reshape(class_probs, (tf.shape(class_probs)[0], -1, tf.shape(class_probs)[-1]))

    return bbox, objectness, class_probs

def draw_outputs(img, boxes, objectness, classes, nums):
    
    print(boxes)
    
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    boxes=np.array(boxes)
    for i in range(nums):
        x1y1 = tuple((boxes[i,0:2] * 128).astype(np.int32))
        x2y2 = tuple((boxes[i,2:4] * 128).astype(np.int32))
        
        print(x1y1)
        print(x2y2)
        
        img = cv2.rectangle(img, (x1y1), (x2y2), (255,0,0), 2)
        
        img = cv2.putText(img, '{} {:.4f}'.format(
            "test", objectness[i]),
                          (x1y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    
    return img


############# DEBUG AND UNUSED FUNCTIONS ##############################

'''
def showDataSet():
	TrainingSet=pd.read_csv(TRAINING_SET_PATH, skiprows=17,delimiter='\s+')
	TrainingSet=TrainingSet[TrainingSet.columns[0:15]]
	TrainingSet.columns=['ID','RA (core)','DEC (core)','RA (centroid)','DEC (centroid)','FLUX','Core frac','BMAJ','BMIN','PA','SIZE','CLASS','SELECTION','x','y']

	print(TrainingSet)
	print(TrainingSet.len())

#given x y pixel coordinates, plot a cutout with x y center
def deb_plot(x, y):
	siz = IMG_SIZE
	fits_img = fits.open(IMG_PATH)
	fits_img = make_fits_2D(fits_img[0])

	WORLD_REF = pywcs.WCS(fits_img.header).deepcopy()

	fits_img = fits_img.data[0, 0]

	pos = (x, y)
	img_fits = Cutout2D(fits_img, position=pos, size=siz,
	                    wcs=WORLD_REF, copy=True)
	
	img_array = img_fits.data

	_, ax = plt.subplots()
	ax.imshow(img_array, cmap='gist_heat')

	plt.show()

	return

def showImage():
	img = mpimg.imread('/data/imageResult1.png')
	imgplot = plt.imshow(img)
	plt.show()

def _setup_pb(primary_beam_file):
    pbhdu = fits.open(primary_beam_file)
    pbhead = pbhdu[0].header
    pb_wcs = pywcs.WCS(pbhead)
    pb_data = pbhdu[0].data[0][0]
    return pb_wcs, pb_data

def train_test_split(filepath, test_size=0.20):
	data = pd.read_csv(filepath)
	print(len(data.img_path))
	file_paths = data.img_path.unique()
	sample_num = int(len(file_paths) * test_size)
	test_paths = np.random.choice(file_paths, size=sample_num)
	print(len(test_paths))
	print(len(file_paths))

'''
