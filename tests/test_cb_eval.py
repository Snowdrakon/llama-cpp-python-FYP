import ctypes

import pytest

import llama_cpp


def test_cb_eval_signature():
    """Ensure the cb_eval callback parameter accepts the full C signature.

    The underlying C type is:
        typedef bool (*ggml_backend_sched_eval_callback)(
            struct ggml_tensor * t,
            bool ask,
            void * user_data);

    Python should allow a callable with three arguments and convert it to the
    proper ctypes CFUNCTYPE.  Previously the type hint only included two
    parameters which would break the conversion.
    """

    called = False

    def callback(tensor_ptr, ask, user_data):
        nonlocal called
        called = True
        # just return True so the scheduler continues
        return True

    params = llama_cpp.llama_context_params()
    # assignment should not raise, and the field should become a CFUNCTYPE
    params.cb_eval = callback
    assert isinstance(params.cb_eval, llama_cpp.ggml_backend_sched_eval_callback)

    # user_data is not set by default, but we can set it and read it back
    dummy = ctypes.c_void_p(1234)
    params.cb_eval_user_data = dummy
    assert params.cb_eval_user_data.value == 1234

    # we can't easily invoke the C-side callback here, but at least the
    # python wrapper respects the signature.

    # also check that the Llama.__init__ annotation was updated
    sig = pytest.importorskip("inspect").signature(llama_cpp.Llama.__init__)
    cb_param = sig.parameters.get("cb_eval")
    assert cb_param is not None
    # ensure the type annotation contains three args if available
    ann = cb_param.annotation
    # the annotation might be Optional[...] so we convert to string
    assert "ctypes.c_void_p" in str(ann)
    assert "," in str(ann), "expected multiple parameters in annotation"
