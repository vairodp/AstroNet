from configs.train_config import IMG_SIZE
from utils import bbox_to_x1y1x2y2

CLASS_LABELS = {
    0 : 'SS-AGN',
    1 : 'FS-AGN',
    2 : 'SFG' 
}

def true_label_to_file(file_path, true_bbox, true_class):
    # true_boxes are in format [x, y, width, height]
    box = bbox_to_x1y1x2y2(true_bbox)
    box *= IMG_SIZE
    left = box[..., 0]
    bottom = box[..., 1]
    right = box[..., 2]
    top = box[..., 3]
    #true_class = tf.map_fn(lambda x: 'SS-AGN' if x == 0 else 'FS-AGN'  if x==1 else 'SFG', true_class, dtype=tf.string)
    with open(file_path, 'w') as outfile:
        for class_id, l, b, r, t in zip(true_class.numpy(), left.numpy(), bottom.numpy(), right.numpy(), top.numpy()):
            outfile.write(CLASS_LABELS[class_id] + f' {round(l)} {round(t)} {round(r)} {round(b)}\n')

def predictions_to_file(file_path, pred_boxes, pred_classes, pred_confs):
    box = pred_boxes * IMG_SIZE
    left = box[..., 0]
    bottom = box[..., 1]
    right = box[..., 2]
    top = box[..., 3]
    with open(file_path, 'w') as outfile:
        outfile.write('PROVA')
        for class_id, conf, l, b, r, t in zip(pred_classes.numpy(), pred_confs.numpy(), left.numpy(), bottom.numpy(), right.numpy(), top.numpy()):
            outfile.write(CLASS_LABELS[class_id] + f' {conf} {round(l)} {round(t)} {round(r)} {round(b)}\n')