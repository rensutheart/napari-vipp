"""Observe attributable RL-family allocations without changing the allocator."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from unittest.mock import patch

from napari_vipp.core.gpu.cupy_runtime import _allocation_pool_owner


@dataclass
class PrivateFFTAllocationTrace:
    allocations: list[tuple[int, int, int]] = field(default_factory=list)
    fft_workspaces: list[tuple[int, bool]] = field(default_factory=list)

    @property
    def managed_peak_bytes(self) -> int:
        return max((reserved for _, reserved, _ in self.allocations), default=0)

    def assert_fft_workspace_ownership(self) -> None:
        assert self.fft_workspaces, "The real FFT path did not expose any plans."
        nonempty = [owned for size, owned in self.fft_workspaces if size]
        assert all(nonempty), "An FFT workspace escaped the runtime's private pool."


@contextmanager
def trace_private_fft_allocations(cupy, runtime):
    """Trace actual pool growth and the owner of native cuFFT work areas.

    CuPy 14.1.1 ``cupy/cuda/cufft.pyx`` PlanNd disables cuFFT automatic
    workspace allocation and uses ``memory.alloc`` / ``cufftSetWorkArea``.
    ``cupy/cuda/memory.pyx:alloc`` delegates to the current allocator. VIPP
    installs this private pool through ``using_allocator`` and disables the
    plan cache for its scope. Assert that ownership on the executed FFT path,
    rather than assuming device-wide memGetInfo changes belong to this test.
    Source: https://github.com/cupy/cupy/blob/v14.1.1/cupy/cuda/cufft.pyx
    """

    pool = runtime._active_pool
    assert pool is not None
    owner = runtime._active_pool_owner
    allocator = cupy.cuda.get_allocator()
    fft_module = importlib.import_module("cupy.fft._fft")
    create_plan = fft_module._get_cufft_plan_nd
    trace = PrivateFFTAllocationTrace()

    def allocate(size):
        pointer = allocator(size)
        trace.allocations.append((int(size), pool.total_bytes(), pool.used_bytes()))
        return pointer

    def create_traced_plan(*args, **kwargs):
        plan = create_plan(*args, **kwargs)
        memory = plan.work_area.mem
        trace.fft_workspaces.append(
            (int(memory.size), _allocation_pool_owner(memory) is owner)
        )
        return plan

    with (
        cupy.cuda.using_allocator(allocate),
        patch.object(fft_module, "_get_cufft_plan_nd", create_traced_plan),
    ):
        yield trace
