"""
Data transformation utilities for bounding box format conversions.
"""


def yolo_to_coco(yolo_bbox, img_width, img_height):
    """
    Converts a YOLO format bounding box to COCO format.

    Args:
        yolo_bbox: [x_center, y_center, width, height] (normalized 0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        [xmin, ymin, xmax, ymax] (absolute pixel values)
    """
    x_center, y_center, width, height = yolo_bbox
    xmin = (x_center - width / 2) * img_width
    ymin = (y_center - height / 2) * img_height
    xmax = (x_center + width / 2) * img_width
    ymax = (y_center + height / 2) * img_height
    return [xmin, ymin, xmax, ymax]


def coco_to_yolo(coco_bbox, img_width, img_height):
    """
    Converts a COCO format bounding box to YOLO format.

    Args:
        coco_bbox: [xmin, ymin, xmax, ymax] (absolute pixel values)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        [x_center, y_center, width, height] (normalized 0-1)
    """
    xmin, ymin, xmax, ymax = coco_bbox
    x_center = ((xmin + xmax) / 2) / img_width
    y_center = ((ymin + ymax) / 2) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return [x_center, y_center, width, height]


__all__ = ['yolo_to_coco', 'coco_to_yolo']
