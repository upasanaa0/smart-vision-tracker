import numpy as np
from collections import OrderedDict


class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=80):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    def update(self, rects):
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = np.array([
            [(x1 + x2) // 2, (y1 + y2) // 2]
            for (x1, y1, x2, y2) in rects
        ])

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
        else:
            ids = list(self.objects.keys())
            existing = np.array(list(self.objects.values()))

            D = np.linalg.norm(existing[:, None] - input_centroids[None, :], axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                obj_id = ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.disappeared[obj_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            for row in set(range(len(ids))) - used_rows:
                obj_id = ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            for col in set(range(len(input_centroids))) - used_cols:
                self.register(input_centroids[col])

        return self.objects