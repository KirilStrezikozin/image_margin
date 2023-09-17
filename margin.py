import bpy
import time
import ctypes
import numpy as np
import multiprocessing

MAX_WORKERS = multiprocessing.cpu_count()


class MarginManager():
    def __init__(self, image: bpy.types.Image, max_workers=MAX_WORKERS
                 ) -> None:
        self.timer = time.perf_counter()
        self.max_workers = max(1, min(max_workers, 61))
        self.workers_count = multiprocessing.Value(ctypes.c_int8, 0)

        self.image = image
        width, height = self.image.size
        self.pixels_size = width * height * 4
        self.rgba_stacks = self.pixels_size // 4

        pixels_buf = multiprocessing.RawArray(ctypes.c_float, self.pixels_size)
        self.pixels = np.ndarray((self.pixels_size,), dtype=np.float32,
                                 buffer=pixels_buf)

        self.image.pixels.foreach_get(self.pixels)

        modified_pixels_buf = multiprocessing.RawArray(
            ctypes.c_bool, self.rgba_stacks)
        self.modified_pixels = np.ndarray(
            (self.rgba_stacks,), dtype=np.bool_, buffer=modified_pixels_buf)

        row = width * 4
        if row + 4 > 32_767:
            row_dtype = np.int32
        else:
            row_dtype = np.int16
        if self.pixels_size > 2_147_483_647:
            i_dtype = np.int64
        elif self.pixels_size > 32_767:
            i_dtype = np.int32
        else:
            i_dtype = np.int16

        self.steps = np.array([
            row,
            4,
            -row,
            -4,
            row + 4,
            -row + 4,
            -row - 4,
            row - 4
        ], dtype=row_dtype)

        self.m_steps = set(self.steps[0:4])

        rays_n = max(width, height)
        rays_m = np.broadcast_to(np.arange(rays_n)[:, np.newaxis], (rays_n, 4))
        self.rays_cast = np.reshape(np.concatenate(
            (rays_m + 1, rays_m), axis=1, dtype=i_dtype) * self.steps, -1)

        self.workers = [None] * max_workers

        msg = ("Infinite margin initialization for "
               + f"{self.image.name} finished in "
               + "%.2f" % (time.perf_counter() - self.timer))
        print(msg)

        self.timer = time.perf_counter()

    def start(self) -> None:
        self.timer = time.perf_counter()
        worker_section = self.pixels_size // self.max_workers

        for worker_i in range(0, self.max_workers):
            start_pixel_i = worker_i * worker_section
            if worker_i + 1 == self.max_workers:
                stop_pixel_i = self.pixels_size
            else:
                stop_pixel_i = (worker_i + 1) * worker_section

            worker = multiprocessing.Process(
                target=self._margin,
                args=[worker_i, start_pixel_i, stop_pixel_i])
            worker.start()

            self.workers[worker_i] = worker

        msg = ("Infinite margin for "
               + f"{self.image.name} started process in "
               + "%.2f" % (time.perf_counter() - self.timer))
        print(msg)

    def join(self) -> None:
        for worker in self.workers:
            if worker is None:
                continue
            worker.join()

        self.image.pixels.foreach_set(self.pixels)

        msg = ("Infinite margin for "
               + f"{self.image.name} finished in "
               + "%.2f" % (time.perf_counter() - self.timer))
        print(msg)

        self.timer = time.perf_counter()

    def is_alive(self) -> bool:
        return any(worker.is_alive() for worker in self.workers)

    def _margin(self, worker_i: int, start_pixel_i: int, stop_pixel_i: int,
                worker_timer: int = 0) -> None:
        requery = False
        if worker_timer == 0:
            worker_timer = time.perf_counter()

        rays = self.rays_cast
        last_i_0a = 0

        for i_0a in range(start_pixel_i + 3, stop_pixel_i, 4):
            if self.pixels[i_0a] != 0:
                self.pixels[i_0a] = 1.0
                continue

            rays += i_0a - last_i_0a
            last_i_0a = i_0a
            v_rays_i = np.where((0 <= rays) & (rays < self.pixels_size))[0]
            v_rays_i = v_rays_i[np.logical_and(
                self.pixels[rays[v_rays_i]] != 0,
                ~self.modified_pixels[rays[v_rays_i] // 4])]

            if v_rays_i.size == 0:
                requery = True
                continue

            i_0 = i_0a - 3
            d = v_rays_i[0]
            step = self.steps[d % 8]
            i_color = rays[d] - 3

            if step not in self.m_steps:
                v_colors = np.where(self.pixels[i_0a:rays[d]:step] != 0)[0]
                if v_colors.size > 0:
                    i_color = i_0 + v_colors[0] * step

            self.pixels.reshape((self.rgba_stacks, 4))[
                i_0 // 4:i_color // 4:step // 4] = self.pixels[
                    i_color:i_color + 4]
            self.modified_pixels[i_0 // 4:i_color // 4:step // 4] = True

        # not all transparent pixels were filled
        if requery:
            self._margin(worker_i, start_pixel_i, stop_pixel_i,
                         worker_timer=worker_timer)

        self.workers_count.value += 1
        msg = ("Infinite margin for "
               + f"{self.image.name}: {self.workers_count.value}/"
               + f"{self.max_workers} workers finished. Last one in "
               + "%.2f" % (time.perf_counter() - worker_timer))
        print(msg)


def image_add_infinite_margin(image: bpy.types.Image) -> None:
    manager = MarginManager(image)
    manager.start()
    manager.join()

    # for Modal Operators:
    # if not manager.is_alive():
    #     manager.join()


def main() -> None:
    image = bpy.data.images['Cone.png']
    image_add_infinite_margin(image)


if __name__ == '__main__':
    main()
