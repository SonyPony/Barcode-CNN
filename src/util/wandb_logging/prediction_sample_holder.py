# coding=utf-8
class PredictionSampleHolder:
    def __init__(self):
        self.image, self.gt_bbox, self.pred_bbox, self.gt_label, self.pred_label = (None, ) * 5

    def __call__(self, i, pred_bbox, pred_label, gt_label, gt_bbox, image):
        if i:
            return
        self.image = image
        self.pred_bbox = pred_bbox
        self.pred_label = pred_label
        self.gt_label = gt_label
        self.gt_bbox = gt_bbox

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        i = self._i
        if i >= self.image.size()[0]:
            raise StopIteration
        self._i += 1
        return self.image[i], self.pred_bbox[i], self.pred_label[i], self.gt_bbox[i], self.gt_label[i]