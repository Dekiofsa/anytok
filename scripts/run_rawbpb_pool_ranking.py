#!/usr/bin/env python3
"""Monitor-only observer for the raw-BPB frozen full-pool ranking Slurm jobs.

Polls the cluster over the user's existing SSH control connection, writes
<local-state>/status.json (atomic) and <local-state>/events.jsonl for the
dashboard (plans/run-dashboard.json, adapter progress_json), and copies the
small result files home when the evaluation is complete.

It never submits, cancels or changes anything on the cluster and never starts
an SSH master connection (BatchMode, existing ControlPath only).

Exit codes: 0 reached --until target, 2 failed, 3 observer lost the cluster.
Python 3.9 compatible, standard library only.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PLATFORM = "Tillicum · H200"
PROFILE = "stf"
TOTAL = 1000
SHARD_SIZE = 250
N_WORKERS = 4
CHECKPOINT_STEP = 20000
EVENT_MAX_AGE = 600.0
SSH_TIMEOUT = 120
SCP_TIMEOUT = 900
MAX_COPY_FAILURES = 5
STATUS_SCRIPT = "code/rawbpb_pool_ranking_status.py"

FAIL_PREFIXES = ("FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL",
                 "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT")
ACTIVE_PREFIXES = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
                   "REQUEUE", "RESIZING", "SIGNALING", "STAGE_OUT", "STOPPED")

# Required files fail the copy when missing; optional ones only produce a warning
# (names come from the cluster-side status script / aggregator, not verified here).
REQUIRED_FILES = ["all-scores.json.gz"]
OPTIONAL_FILES = ["policy.json", "panel-manifest.json", "run-plan.json",
                  "checkpoint-manifest.json", "preparation-status.json"]
STATE_FILES = REQUIRED_FILES + OPTIONAL_FILES

OPTS = {"control_path": None}


class SSHError(RuntimeError):
    """ssh itself failed (rc 255 or timeout): connectivity, not the remote command."""


class StatusError(RuntimeError):
    """The remote status command ran but failed or printed something unusable."""


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ssh_options():
    return ["-o", "ControlPath=%s" % OPTS["control_path"],
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]


def ssh_run(host, command, timeout=SSH_TIMEOUT):
    """Run one remote command. Returns (returncode, stdout, stderr); rc 255 = ssh failure."""
    argv = ["ssh"] + ssh_options() + [host, command]
    try:
        p = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=timeout,
                           universal_newlines=True)
    except subprocess.TimeoutExpired:
        return 255, "", "ssh timeout after %ss" % timeout
    except OSError as exc:
        return 255, "", "ssh could not start: %s" % exc
    return p.returncode, p.stdout, p.stderr


def scp_fetch(host, remote_path, local_path, timeout=SCP_TIMEOUT):
    argv = ["scp", "-q"] + ssh_options() + ["%s:%s" % (host, remote_path), str(local_path)]
    try:
        p = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=timeout,
                           universal_newlines=True)
    except subprocess.TimeoutExpired:
        return 255, "scp timeout after %ss" % timeout
    except OSError as exc:
        return 255, "scp could not start: %s" % exc
    return p.returncode, p.stderr


def dq(text):
    """Double-quote for the remote shell but let $VARS (e.g. $USER) expand there."""
    return '"%s"' % re.sub(r'(["\\`])', r"\\\1", text)


def parse_elapsed(text):
    """[D-]HH:MM:SS, HH:MM:SS or MM:SS -> seconds (None if unparsable)."""
    try:
        days = 0
        text = text.strip()
        if "-" in text:
            d, text = text.split("-", 1)
            days = int(d)
        parts = [float(x) for x in text.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        return int(days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2])
    except (ValueError, AttributeError):
        return None


def atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with open(str(tmp), "w") as handle:
        json.dump(obj, handle, indent=2, sort_keys=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(tmp), str(path))


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_fail(state):
    return bool(state) and state.upper().startswith(FAIL_PREFIXES)


def is_active(state):
    return bool(state) and state.upper().startswith(ACTIVE_PREFIXES)


def is_completed(state):
    return bool(state) and state.upper().startswith("COMPLETED")


class Observer:
    def __init__(self, args):
        self.a = args
        self.local = Path(args.local_state)
        self.local.mkdir(parents=True, exist_ok=True)
        self.status_path = self.local / "status.json"
        self.events_path = self.local / "events.jsonl"
        self.ssh_failures = 0
        self.status_errors = 0
        self.copy_failures = 0
        self.score_done_polls = 0
        self.last_status = {}
        self.last_event_sig = None
        self.last_event_time = 0.0
        self.started_at = utc_now()
        try:
            prev = json.loads(self.status_path.read_text())
            if isinstance(prev.get("observer_started_at"), str):
                self.started_at = prev["observer_started_at"]
        except (OSError, ValueError):
            pass

    # ---------------------------------------------------------------- cluster
    def fetch_status(self):
        """Return the status dict, or None while STATE or the status script does not exist yet."""
        py, state = self.a.python, self.a.remote_state
        script = state.rstrip("/") + "/" + STATUS_SCRIPT
        cmd = ('if [ -d %s ] && [ -f %s ]; then %s %s --state %s; else echo @@MISSING; fi'
               % (dq(state), dq(script), dq(py), dq(script), dq(state)))
        rc, out, err = ssh_run(self.a.host, cmd)
        if rc == 255:
            raise SSHError("ssh failure: %s" % (err or out).strip()[:300])
        lines = [ln for ln in out.splitlines() if ln.strip()]
        if lines and lines[-1].strip() == "@@MISSING":
            return None
        if rc != 0:
            raise StatusError("status command rc=%s: %s" % (rc, (err or out).strip()[:300]))
        if not lines:
            raise StatusError("status command printed nothing")
        try:
            data = json.loads(lines[-1])
        except ValueError as exc:
            raise StatusError("status command output is not JSON: %s" % exc)
        if not isinstance(data, dict):
            raise StatusError("status command output is not an object")
        return data

    def query_queue(self, job_ids):
        """squeue for each id -> {job_id: [(JobID, state, elapsed, reason)]}."""
        cmd = "; ".join("echo @@%s; squeue -h -r -j %s -o '%%i|%%T|%%M|%%R' 2>/dev/null"
                        % (shlex.quote(j), shlex.quote(j)) for j in job_ids)
        rc, out, err = ssh_run(self.a.host, cmd)
        if rc == 255:
            raise SSHError("squeue ssh failure: %s" % err.strip()[:300])
        rows = {j: [] for j in job_ids}
        cur = None
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("@@"):
                cur = line[2:]
                continue
            parts = line.split("|", 3)
            if cur in rows and len(parts) == 4:
                rows[cur].append((parts[0], parts[1], parts[2], parts[3]))
        return rows

    def query_sacct(self, job_id):
        cmd = "sacct -n -P -j %s --format=JobID,State,ExitCode,Elapsed 2>/dev/null" % shlex.quote(job_id)
        rc, out, err = ssh_run(self.a.host, cmd)
        if rc == 255:
            raise SSHError("sacct ssh failure: %s" % err.strip()[:300])
        rows = []
        for line in out.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 4:
                continue
            jid = parts[0]
            if "." in jid:  # .batch / .extern / numbered steps
                continue
            rows.append((jid, parts[1], parts[2], parts[3]))
        return rows

    def job_table(self, submissions):
        """Return {unit_id: dict(state, reason, elapsed, exit_code, source)} for prep and score tasks."""
        prep = str(submissions["prep"]) if submissions.get("prep") else None
        score = str(submissions["score"]) if submissions.get("score") else None
        units = []  # (unit_id, base_job_id)
        if prep:
            units.append((prep, prep))
        if score:
            units.extend(("%s_%d" % (score, k), score) for k in range(N_WORKERS))
        base_ids = [i for i in (prep, score) if i]
        queue = self.query_queue(base_ids) if base_ids else {}
        table = {}
        need_sacct = set()
        for unit, base in units:
            rows = queue.get(base, [])
            hit = [r for r in rows if r[0] == unit]
            if not hit:  # an unexpanded pending array looks like "<id>_[0-3]"
                hit = [r for r in rows if base != unit and r[0].startswith(base + "_[")]
            if hit:
                jid, state, elapsed, reason = hit[0]
                table[unit] = {"state": state, "reason": reason,
                               "elapsed_seconds": parse_elapsed(elapsed),
                               "exit_code": None, "source": "squeue"}
            else:
                need_sacct.add(base)
        acct = {}
        for base in sorted(need_sacct):
            acct[base] = self.query_sacct(base)
        for unit, base in units:
            if unit in table:
                continue
            rows = acct.get(base, [])
            hit = [r for r in rows if r[0] == unit]
            if not hit and base != unit:
                hit = [r for r in rows if r[0].startswith(base + "_[")]
            if hit:
                jid, state, exit_code, elapsed = hit[0]
                table[unit] = {"state": state, "reason": "", "elapsed_seconds": parse_elapsed(elapsed),
                               "exit_code": exit_code, "source": "sacct"}
            else:
                table[unit] = {"state": "UNKNOWN", "reason": "not in squeue or sacct yet",
                               "elapsed_seconds": None, "exit_code": None, "source": "none"}
        return units, table

    # ------------------------------------------------------------- composition
    def build_workers(self, status, submissions, units, table):
        workers = []
        prep = str(submissions["prep"]) if submissions.get("prep") else None
        score = str(submissions["score"]) if submissions.get("score") else None
        pstat = status.get("preparation") or {}
        wstat = status.get("workers") or {}
        etas = []
        for unit, base in units:
            job = table[unit]
            slurm = job["state"]
            reason = job["reason"]
            entry = {"id": unit, "profile": PROFILE, "platform": PLATFORM}
            if unit == prep:
                if is_active(slurm) and slurm.upper().startswith("RUNNING") and pstat.get("stage"):
                    text = "Preparing · %s" % pstat.get("stage")
                    if pstat.get("detail"):
                        text += " · %s" % pstat["detail"]
                    entry["state"] = text
                else:
                    entry["state"] = "%s%s" % (slurm, " (%s)" % reason if reason else "")
                if pstat.get("completed") is not None and pstat.get("total"):
                    entry["task"] = "%s/%s %s" % (pstat.get("completed"), pstat.get("total"),
                                                  pstat.get("stage", ""))
                else:
                    entry["task"] = "preparation: %s" % pstat.get("stage", "waiting")
                entry["elapsed_seconds"] = job["elapsed_seconds"]
            else:
                k = unit.rsplit("_", 1)[-1]
                w = wstat.get(k) or {}
                wstage = w.get("stage")
                done = w.get("completed") if wstage in ("scoring", "complete", "loading_frozen_checkpoint") else None
                if is_completed(slurm) and wstage == "complete":
                    done = w.get("completed", SHARD_SIZE)
                done = int(done) if isinstance(done, (int, float)) else 0
                running = bool(slurm) and slurm.upper().startswith("RUNNING")
                if running and wstage == "scoring" and w.get("index") is not None:
                    entry["state"] = "Tokenizer #%s · batch %s/%s" % (
                        w.get("index"), w.get("batches_scored", 0), w.get("batch_total", 32))
                elif running and wstage == "loading_frozen_checkpoint":
                    entry["state"] = "RUNNING · loading frozen checkpoint"
                else:
                    entry["state"] = "%s%s" % (slurm, " (%s)" % reason if reason else "")
                entry["task"] = "%d/%d complete" % (done, w.get("total", SHARD_SIZE))
                el = w.get("elapsed_seconds") if running and isinstance(w.get("elapsed_seconds"), (int, float)) else None
                if el is None:
                    el = job["elapsed_seconds"]
                entry["elapsed_seconds"] = el
                resumed = w.get("resumed", w.get("resumed_count", w.get("resumed_scores")))
                fresh = done - int(resumed) if isinstance(resumed, (int, float)) else done
                if running and fresh > 0 and el:
                    eta = int(el / fresh * max(0, w.get("total", SHARD_SIZE) - done))
                    entry["eta_seconds"] = eta
                    etas.append(eta)
            entry["slurm_state"] = slurm
            if job["exit_code"]:
                entry["exit_code"] = job["exit_code"]
            workers.append(entry)
        return workers, (max(etas) if etas else None)

    def failure_text(self, units, table):
        bad = []
        for unit, _base in units:
            j = table[unit]
            if is_fail(j["state"]):
                bad.append("%s %s (exit %s, elapsed %ss)" % (unit, j["state"], j["exit_code"],
                                                            j["elapsed_seconds"]))
        return "; ".join(bad)

    def logs(self, submissions):
        base = self.a.remote_state.rstrip("/")
        out = []
        if submissions.get("prep"):
            out.append("%s/slurm-prep-%s.log" % (base, submissions["prep"]))
        if submissions.get("score"):
            out.extend("%s/slurm-%s-%d.log" % (base, submissions["score"], k) for k in range(N_WORKERS))
        return out

    # ------------------------------------------------------------------ output
    def emit(self, record, now):
        record["at_utc"] = utc_now()
        record.setdefault("total", TOTAL)
        record.setdefault("platform", PLATFORM)
        record.setdefault("profile", PROFILE)
        record.setdefault("checkpoint_step", CHECKPOINT_STEP)
        record["observer_started_at"] = self.started_at
        try:
            record["elapsed_seconds"] = max(0, int((datetime.now(timezone.utc) -
                                                    datetime.fromisoformat(self.started_at)).total_seconds()))
        except ValueError:
            pass
        self.last_status = record
        atomic_write_json(self.status_path, record)
        sig = (record.get("stage"), record.get("completed"),
               tuple((w["id"], w.get("slurm_state"), w.get("task")) for w in record.get("workers", [])),
               record.get("state"))
        if sig != self.last_event_sig or now - self.last_event_time >= EVENT_MAX_AGE:
            ev = {"at_utc": record["at_utc"], "stage": record.get("stage"),
                  "completed": record.get("completed"), "total": record["total"]}
            for key in ("state", "error", "eta_seconds", "detail"):
                if record.get(key) is not None:
                    ev[key] = record[key]
            ev["worker_states"] = {w["id"]: w.get("slurm_state") for w in record.get("workers", [])}
            with open(str(self.events_path), "a") as handle:
                handle.write(json.dumps(ev) + "\n")
                handle.flush()
            self.last_event_sig = sig
            self.last_event_time = now
        line = {"at_utc": record["at_utc"], "stage": record.get("stage"),
                "completed": record.get("completed"), "eta_seconds": record.get("eta_seconds"),
                "ssh_failures": self.ssh_failures}
        if record.get("error"):
            line["error"] = record["error"]
        print(json.dumps(line), flush=True)

    # -------------------------------------------------------------------- copy
    def resolve_remote(self, path):
        rc, out, err = ssh_run(self.a.host, "printf '%%s' %s" % dq(path))
        if rc != 0 or not out.strip():
            raise RuntimeError("could not resolve remote path %s: %s" % (path, err.strip()[:200]))
        return out.strip()

    def copy_results(self):
        state = self.resolve_remote(self.a.remote_state.rstrip("/"))
        inp = self.resolve_remote(self.a.remote_input.rstrip("/"))
        pairs = [("%s/%s" % (state, n), n) for n in STATE_FILES]
        pairs.append(("%s/pool/full-results.json" % inp, "pool-index.json"))
        rc, out, err = ssh_run(self.a.host, "sha256sum %s 2>/dev/null; true"
                               % " ".join(shlex.quote(r) for r, _ in pairs))
        if rc == 255:
            raise SSHError("remote sha256sum ssh failure: %s" % err.strip()[:300])
        remote_hash = {}
        for line in out.splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                remote_hash[parts[1].strip().lstrip("*")] = parts[0]
        copied = {}
        warnings = self.copy_warnings = []
        for remote, name in pairs:
            want = remote_hash.get(remote)
            if not want:
                if name in OPTIONAL_FILES:
                    warnings.append("optional file %s missing on the cluster; not copied" % name)
                    continue
                raise RuntimeError("no remote sha256 for required file %s" % remote)
            tmp = self.local / (".%s.part" % name)
            rc, err = scp_fetch(self.a.host, remote, tmp)
            if rc != 0:
                raise RuntimeError("scp %s failed: %s" % (remote, err.strip()[:300]))
            got = sha256_file(tmp)
            if got != want:
                try:
                    os.remove(str(tmp))
                except OSError:
                    pass
                raise RuntimeError("sha256 mismatch for %s: remote %s local %s" % (name, want, got))
            os.replace(str(tmp), str(self.local / name))
            copied[name] = want
        return copied

    # -------------------------------------------------------------------- poll
    def poll(self, now):
        """One poll. Returns an exit code, or None to continue."""
        try:
            status = self.fetch_status()
            if status is None:
                self.ssh_failures = 0
                self.status_errors = 0
                rec = {"completed": 0, "preparation": None, "submissions": None,
                       "checkpoint_step": CHECKPOINT_STEP, "remote_logs": [], "ssh_failures": 0,
                       "stage": "waiting_for_submission", "workers": [], "worker_jobs": {},
                       "detail": "remote state directory or status script not staged yet"}
                self.emit(rec, now)
                return None
            submissions = status.get("submissions") or {}
            units, table = self.job_table(submissions)
            if any(not is_active(table[u]["state"]) and table[u]["state"] != "UNKNOWN" for u, _ in units):
                refreshed = self.fetch_status()  # a job just ended: refresh so the files are not stale
                if refreshed is not None:
                    status = refreshed
                    submissions = status.get("submissions") or submissions
                    units, table = self.job_table(submissions)
        except (SSHError, StatusError) as exc:
            if isinstance(exc, SSHError):
                self.ssh_failures += 1
            else:
                self.status_errors += 1
            rec = dict(self.last_status)
            rec["ssh_failures"] = self.ssh_failures
            rec["status_errors"] = self.status_errors
            rec["last_error"] = str(exc)
            if self.ssh_failures >= self.a.max_ssh_failures or self.status_errors >= self.a.max_ssh_failures:
                if isinstance(exc, SSHError):
                    why = "%d consecutive SSH failures" % self.ssh_failures
                else:
                    why = "the remote status command failed %d times in a row" % self.status_errors
                rec["stage"] = "observer_lost"
                rec["state"] = "stopped"
                rec.pop("eta_seconds", None)
                rec["error"] = ("observer lost the cluster (%s); the Slurm jobs are not affected: %s"
                                % (why, exc))
                self.emit(rec, now)
                return 3
            rec.setdefault("stage", "waiting_for_submission")
            rec.setdefault("completed", 0)
            self.emit(rec, now)
            return None
        self.ssh_failures = 0
        self.status_errors = 0

        prep_status = status.get("preparation") or {}
        wstat = status.get("workers") or {}
        completed = status.get("completed")
        completed = max(0, min(TOTAL, int(completed))) if isinstance(completed, (int, float)) else 0
        rec = {"completed": completed, "preparation": status.get("preparation"),
               "submissions": submissions or None, "checkpoint_step": CHECKPOINT_STEP,
               "remote_logs": self.logs(submissions), "ssh_failures": 0}
        if not submissions.get("prep"):
            rec.update(stage="waiting_for_submission", workers=[], worker_jobs={},
                       detail="no Slurm submissions recorded yet")
            self.emit(rec, now)
            return None
        workers, eta = self.build_workers(status, submissions, units, table)
        rec["workers"] = workers
        rec["worker_jobs"] = {u: {"state": table[u]["state"], "reason": table[u]["reason"],
                                  "exit_code": table[u]["exit_code"],
                                  "elapsed_seconds": table[u]["elapsed_seconds"]} for u, _ in units}
        if eta is not None:
            rec["eta_seconds"] = eta
        prep_id = str(submissions["prep"])
        score_id = str(submissions["score"]) if submissions.get("score") else None
        prep_state = table[prep_id]["state"]

        def fail(message):
            rec.update(stage="failed", state="failed", error=message)
            rec.pop("eta_seconds", None)
            self.emit(rec, now)
            return 2

        bad = self.failure_text(units, table)
        if bad:
            return fail("Slurm job failed: %s" % bad)
        if prep_status.get("stage") == "failed":
            return fail("preparation failed: %s" % (prep_status.get("error") or prep_status.get("detail")))

        # preparation phase
        if is_active(prep_state) or prep_state == "UNKNOWN":
            rec.update(stage="preparing",
                       detail="%s%s" % (prep_status.get("stage", "queued"),
                                        (" · " + prep_status["detail"]) if prep_status.get("detail") else ""))
            self.emit(rec, now)
            return None
        w0 = wstat.get("0") or {}
        smoke_ok = (w0.get("stage") == "smoke_stop" and isinstance(w0.get("completed"), (int, float))
                    and w0["completed"] >= 1)
        prepared_ok = (is_completed(prep_state) and prep_status.get("stage") == "prepared"
                       and (smoke_ok or score_id is not None))
        if not prepared_ok:
            return fail("prep job %s ended (%s) without a verified preparation: preparation stage %r, "
                        "worker 0 stage %r completed %r"
                        % (prep_id, prep_state, prep_status.get("stage"), w0.get("stage"), w0.get("completed")))
        if score_id is None:
            rec.update(stage="prepared", detail="inputs prepared and smoke test passed; scoring not submitted")
            self.emit(rec, now)
            return 0 if self.a.until == "prepared" else None
        if self.a.until == "prepared":
            rec.update(stage="prepared", detail="inputs prepared; scoring already submitted")
            self.emit(rec, now)
            return 0

        # scoring phase
        if status.get("complete"):
            rec.update(stage="copying_results", detail="copying results from the cluster")
            try:
                copied = self.copy_results()
            except (RuntimeError, OSError) as exc:
                self.copy_failures += 1
                rec["last_error"] = str(exc)
                if self.copy_failures >= MAX_COPY_FAILURES:
                    return fail("copying results failed %d times: %s" % (self.copy_failures, exc))
                self.emit(rec, now)
                return None
            if self.copy_warnings:
                rec["warnings"] = list(self.copy_warnings)
            rec.update(stage="complete", completed=TOTAL, copied_files=copied,
                       detail="all %d tokenizers scored; results copied and verified" % TOTAL)
            rec.pop("eta_seconds", None)
            self.emit(rec, now)
            return 0
        task_states = [table["%s_%d" % (score_id, k)]["state"] for k in range(N_WORKERS)]
        if all(is_completed(s) for s in task_states):
            self.score_done_polls += 1
            if self.score_done_polls >= 2:
                return fail("all score tasks COMPLETED but the status reports %d/%d scores or incomplete workers"
                            % (completed, TOTAL))
        else:
            self.score_done_polls = 0
        rec.update(stage="scoring", detail="%d/%d tokenizers scored" % (completed, TOTAL))
        self.emit(rec, now)
        return None


def parse_args(argv=None):
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--host", required=True)
    p.add_argument("--control-path", default="~/.ssh/cm-tillicum-*",
                   help="SSH ControlPath socket; the default glob must match exactly one socket")
    p.add_argument("--remote-state", required=True)
    p.add_argument("--remote-input", required=True)
    p.add_argument("--python", required=True, help="remote python executable")
    p.add_argument("--local-state", default=str(root / "modal_results" / "rawbpb_pool_ranking_20000"))
    p.add_argument("--until", required=True, choices=["prepared", "complete"])
    p.add_argument("--interval", type=float, default=60)
    p.add_argument("--max-ssh-failures", type=int, default=30)
    args = p.parse_args(argv)
    args.control_path = os.path.expanduser(args.control_path)
    if any(ch in args.control_path for ch in "*?["):
        import glob
        import stat

        def is_sock(path):
            try:
                return stat.S_ISSOCK(os.stat(path).st_mode)
            except OSError:
                return False
        found = [f for f in glob.glob(args.control_path) if is_sock(f)]
        if len(found) != 1:
            p.error("--control-path glob %s matched %d sockets; pass the exact path" % (args.control_path, len(found)))
        args.control_path = found[0]
    return args


def main(argv=None):
    args = parse_args(argv)
    OPTS["control_path"] = args.control_path
    obs = Observer(args)
    try:
        while True:
            code = obs.poll(time.time())
            if code is not None:
                return code
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
