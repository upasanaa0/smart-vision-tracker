class ZoneCrossingCounter:
    """
    Counts objects crossing a virtual line defined by two points.
    Uses the sign of the cross product to detect direction of crossing.
    """

    def __init__(self, line_start, line_end):
        self.p1 = line_start   # (x1, y1)
        self.p2 = line_end     # (x2, y2)
        self.prev_positions = {}   # id -> side (-1 or 1)
        self.count_in = 0
        self.count_out = 0

    def _side(self, point):
        """Returns which side of the line the point is on via cross product."""
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        px = point[0] - self.p1[0]
        py = point[1] - self.p1[1]
        cross = dx * py - dy * px
        return 1 if cross >= 0 else -1

    def update(self, track_map):
        """
        track_map: {obj_id: (cx, cy)} from CentroidTracker
        Returns: (count_in, count_out)
        """
        for obj_id, centroid in track_map.items():
            current_side = self._side(centroid)
            if obj_id in self.prev_positions:
                prev_side = self.prev_positions[obj_id]
                if prev_side != current_side:
                    if current_side == 1:
                        self.count_in += 1
                    else:
                        self.count_out += 1
            self.prev_positions[obj_id] = current_side
        return self.count_in, self.count_out

    def draw(self, frame):
        import cv2
        cv2.line(frame, self.p1, self.p2, (0, 255, 255), 2)
        cv2.putText(frame, f"IN:  {self.count_in}",  (self.p1[0], self.p1[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2)
        cv2.putText(frame, f"OUT: {self.count_out}", (self.p1[0], self.p1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)
        return frame