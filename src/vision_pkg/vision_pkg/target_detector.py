'''
[EN]
Target Detect
A module that detects books and converts 
the coordinates of objects detected via camera (e.g., RealSense) 
into robot-compatible X, Y, Z spatial coordinates.

[KR]
책을 디텍팅하는 파일
realsense로 검출한객체의 좌표값을 로봇이 사용 가능한 x,y,z 좌표값으로 반환
'''


class TargetDetector:
    def __init__(
        self,
        shelf_x_min=-0.5,
        shelf_x_max=0.5,
        shelf_y_min=-0.5,
        shelf_y_max=0.5,
        shelf_z=1.0,
        book_width=0.25,
        shelf_levels=None,
        row_tolerance=0.12,
        minimum_gap=0.20,
    ):
        if shelf_x_max <= shelf_x_min:
            raise ValueError('shelf_x_max must be greater than shelf_x_min')
        if book_width <= 0 or minimum_gap <= 0:
            raise ValueError('book_width and minimum_gap must be positive')

        self.shelf_x_min = shelf_x_min
        self.shelf_x_max = shelf_x_max
        self.shelf_y_min = shelf_y_min
        self.shelf_y_max = shelf_y_max
        self.shelf_z = shelf_z
        self.book_width = book_width
        self.shelf_levels = [
            float(level) for level in (shelf_levels or [(shelf_y_min + shelf_y_max) / 2])
        ]
        self.row_tolerance = row_tolerance
        self.minimum_gap = minimum_gap

    def process(self, detected_books):
        """Return the leftmost empty slot, or ``None`` when no slot exists.

        ``detected_books`` is the list returned by ``BookDetector.process``.
        Each item must contain an ``xyz`` tuple and may contain ``box``.
        The returned dictionary contains the slot center and its shelf row.
        """
        candidates = []
        for row_y in self.shelf_levels:
            books = [
                book for book in detected_books
                if abs(float(book['xyz'][1]) - row_y) <= self.row_tolerance
            ]
            row_z = self._row_depth(books)
            occupied = []
            for book in books:
                x = float(book['xyz'][0])
                width = self._book_width(book)
                occupied.append((x - width / 2, x + width / 2))

            occupied.sort()
            cursor = self.shelf_x_min
            for start, end in occupied:
                start = max(self.shelf_x_min, start)
                end = min(self.shelf_x_max, end)
                if start - cursor >= max(self.minimum_gap, self.book_width):
                    candidates.append((cursor + self.book_width / 2, row_y, row_z))
                cursor = max(cursor, end)

            if self.shelf_x_max - cursor >= max(self.minimum_gap, self.book_width):
                candidates.append((cursor + self.book_width / 2, row_y, row_z))

        if not candidates:
            return None

        # Only one target is returned: prioritize the greatest depth, then X.
        x, y, z = max(candidates, key=lambda candidate: (candidate[2], -candidate[0]))
        return self._make_target(x, y, z)

    def _book_width(self, book):
        box = book.get('box')
        if box is None:
            return self.book_width
        return self.book_width

    def _row_depth(self, books):
        if not books:
            return float(self.shelf_z)
        return sum(float(book['xyz'][2]) for book in books) / len(books)

    def _make_target(self, x, y, z):
        return {
            'xyz': (float(x), float(y), float(z)),
            'shelf_row_y': float(y),
            'type': 'empty_shelf_position',
        }