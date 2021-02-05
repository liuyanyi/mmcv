from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['polygon_iou'])


def polygon_iou(poly1, poly2):
    return ext_module.polygon_iou(poly1, poly2)
