import contextlib, io, sys, time, traceback, multiprocessing as mp


class ExecResult:
    def __init__(self, ok: bool, stdout: str, error: str):
        self.ok = ok; self.stdout = stdout; self.error = error


_DEF_TIMEOUT=5


def _run(code: str, q: mp.Queue):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ns = {}
            exec(code, ns, ns)
        q.put(ExecResult(True, buf.getvalue(), ""))
    except Exception as e:
        q.put(ExecResult(False, buf.getvalue(), traceback.format_exc()))


def safe_exec(code: str, timeout: int=_DEF_TIMEOUT) -> ExecResult:
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=_run, args=(code, q))
    p.start(); p.join(timeout)
    if p.is_alive():
        p.terminate(); return ExecResult(False, "", "Timeout")
    return q.get() if not q.empty() else ExecResult(False, "", "NoResult")