"""Recognize real Python draw/upload frames with no Qt window or GL calls."""

from types import SimpleNamespace

import pytest
from napari.utils.notifications import ErrorNotification, notification_manager
from OpenGL.error import GLError
from vispy.gloo import glir
from vispy.scene import SceneCanvas

from napari_vipp.ui.volume_render_status import (
    is_3d_texture_allocation_failure,
    renderer_error_notifications,
    scene_canvas,
)

canvas_module = pytest.importorskip("napari._vispy.canvas")
pytestmark = pytest.mark.skipif(
    not all(
        hasattr(canvas_module, name) for name in ("NapariSceneCanvas", "VispyCanvas")
    ),
    reason="Napari does not expose the recognized VisPy canvas boundary.",
)


def _error_notification(call):
    try:
        call()
    except (GLError, RuntimeError) as error:
        return ErrorNotification(error)
    pytest.fail("Expected the stubbed GL operation to raise.")


def _canvas():
    return object.__new__(canvas_module.NapariSceneCanvas)


@pytest.fixture
def draw(monkeypatch):
    """The real napari draw delegates to a fake scene body, never a backend."""

    def scene_body(self, visual, event=None):
        visual()

    monkeypatch.setattr(SceneCanvas, "draw_visual", scene_body)
    return lambda canvas, visual: canvas.draw_visual(visual)


def _upload(monkeypatch, name, error, target=glir.GL_TEXTURE_3D):
    def fail(*args):
        raise error

    monkeypatch.setattr(
        glir, "_check_pyopengl_3D", lambda: SimpleNamespace(**{name: fail})
    )
    if name == "glTexImage3D":
        return lambda: glir.glTexImage3D(target, 0, 0x8229, 0x1903, 0x1400, (2, 3, 5))
    # The wrapper only needs the shape before the fake driver is called.
    data = SimpleNamespace(shape=(2, 3, 5))
    return lambda: glir.glTexSubImage3D(target, 0, 0, 0, 0, 0x1903, 0x1401, data)


@pytest.mark.parametrize("operation", ("glTexImage3D", "glTexSubImage3D"))
def test_recognizes_exact_canvas_and_deferred_texture_frame(
    draw, monkeypatch, operation
):
    own, unrelated = _canvas(), _canvas()
    upload = _upload(monkeypatch, operation, GLError(err=0x0505))
    event = _error_notification(lambda: draw(own, upload))
    assert is_3d_texture_allocation_failure(event, (None, own))
    assert not is_3d_texture_allocation_failure(event, (unrelated,))
    assert not is_3d_texture_allocation_failure(event, ())
    assert event.exception.err == 0x0505  # No change to the original notification.


def test_ignores_a_nested_unowned_canvas_draw(draw, monkeypatch):
    own, other = _canvas(), _canvas()
    upload = _upload(monkeypatch, "glTexImage3D", GLError(err=0x0505))
    event = _error_notification(lambda: draw(own, lambda: draw(other, upload)))
    assert not is_3d_texture_allocation_failure(event, (own,))
    assert is_3d_texture_allocation_failure(event, (other,))


@pytest.mark.parametrize(
    ("error", "target"),
    [
        (GLError(err=0x0502), glir.GL_TEXTURE_3D),  # Invalid operation is not OOM.
        (GLError(err=0x0505), 0x8070),  # A proxy is not a real texture allocation.
        (RuntimeError("glTexImage3D: GL_OUT_OF_MEMORY"), glir.GL_TEXTURE_3D),
    ],
)
def test_ignores_other_errors_and_proxy_checks(draw, monkeypatch, error, target):
    own = _canvas()
    upload = _upload(monkeypatch, "glTexImage3D", error, target)
    event = _error_notification(lambda: draw(own, upload))
    assert not is_3d_texture_allocation_failure(event, (own,))


def test_names_alone_do_not_establish_a_draw_or_upload(draw, monkeypatch):
    own = _canvas()
    upload = _upload(monkeypatch, "glTexImage3D", GLError(err=0x0505))

    def draw_visual(self):
        upload()

    def glTexImage3D():  # noqa: N802
        raise GLError(err=0x0505)

    no_draw = _error_notification(lambda: draw_visual(own))
    no_upload = _error_notification(lambda: draw(own, glTexImage3D))
    assert not is_3d_texture_allocation_failure(no_draw, (own,))
    assert not is_3d_texture_allocation_failure(no_upload, (own,))
    assert not is_3d_texture_allocation_failure(
        SimpleNamespace(exception=no_upload.exception), (own,)
    )


def test_uses_the_existing_notification_emitter_and_exact_viewer_identity():
    assert renderer_error_notifications() is notification_manager.notification_ready
    model = object()
    renderer = object.__new__(canvas_module.VispyCanvas)
    renderer.viewer = model
    renderer._scene_canvas = _canvas()
    assert (
        scene_canvas(SimpleNamespace(viewer=model, canvas=renderer))
        is renderer._scene_canvas
    )
    assert scene_canvas(SimpleNamespace(viewer=object(), canvas=renderer)) is None
    assert scene_canvas(SimpleNamespace(viewer=model, canvas=object())) is None
    assert scene_canvas(None) is None
