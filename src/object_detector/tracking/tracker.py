from sort_tracker import Sort

def create_tracker(max_age=5, min_hits=1, iou_threshold=0.3):
    return Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
