"""Finish the conflicted merge of the bot fix: keep this PC's own runner.py and add the fix's three
small runner additions to it. Any surprise -> the merge is aborted and nothing changes."""
import os, py_compile, re, subprocess, sys

P = "src/quantlab/pipeline/runner.py"


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True)


def stop(msg):
    print("STOPPED: " + msg)
    print("Undoing the merge (git merge --abort); your files are back as they were.")
    print(git("merge", "--abort").stdout)
    sys.exit(1)


if not os.path.exists(os.path.join(".git", "MERGE_HEAD")):
    print("No merge in progress here: nothing to do.")
    sys.exit(1)
unmerged = [x for x in git("diff", "--name-only", "--diff-filter=U").stdout.split() if x]
if unmerged != [P]:
    stop(f"expected only {P} in conflict, found {unmerged}")
git("checkout", "--ours", P)
s = open(P, encoding="utf-8").read()
lines = s.split("\n")

# 1. settings, after max_job_attempts
idx = [i for i, l in enumerate(lines) if 'self.max_job_attempts = int(g("max_job_attempts"' in l]
if len(idx) != 1:
    stop("could not find the max_job_attempts line exactly once")
ind = lines[idx[0]][: len(lines[idx[0]]) - len(lines[idx[0]].lstrip())]
cfg = ["# Alpaca paper charges regulatory/clearing fees (cents) as FEE activities; the ledger records",
       "# them before each cash check so they never accumulate into a false mismatch. A single fee",
       "# above broker_fee_max_auto is left unbooked: it still mismatches and pauses for review.",
       'self.book_broker_fees = bool(g("book_broker_fees", True))',
       'self.broker_fees_from = g("broker_fees_from", None)',
       'self.broker_fee_max_auto = float(g("broker_fee_max_auto", 5.0))']
lines[idx[0] + 1:idx[0] + 1] = [ind + c for c in cfg]

# 2. the method, before _reconcile_once
idx = [i for i, l in enumerate(lines) if re.match(r"\s*def _reconcile_once\(self", l)]
if len(idx) != 1:
    stop("could not find def _reconcile_once exactly once")
ind = lines[idx[0]][: len(lines[idx[0]]) - len(lines[idx[0]].lstrip())]
method = '''def _book_broker_fees(self) -> None:
    """Record the broker's new FEE activities in the BOT ledger (once each). A fee list the
    broker cannot deliver is a WARN only: the cash comparison that follows decides."""
    if not self.book_broker_fees or self.ledger is None:
        return
    since = str(self.broker_fees_from) if self.broker_fees_from else self.ledger.initialized_on()
    if not since:
        return
    last = self.ledger.latest_broker_fee_date()     # fetch only recent pages; booking stays idempotent
    fetch_from = max(since, (date.fromisoformat(last) - timedelta(days=10)).isoformat()) if last else since
    try:
        fees = self.broker.fee_activities(fetch_from)
    except BrokerError as exc:
        self.event("WARN", "reconcile", f"broker fee activities unavailable: {exc}"[:300])
        return
    res = self.ledger.book_broker_fees(fees, since=since, max_each=self.broker_fee_max_auto)
    if res["booked"]:
        total = sum(f["net_amount"] for f in res["booked"])
        self.event("INFO", "reconcile", f"booked {len(res['booked'])} broker fee(s), {total:+.2f} USD, "
                   f"into the {BOOK} ledger", res)
    if res["too_large"]:
        self.event("WARN", "reconcile", f"{len(res['too_large'])} broker fee(s) above "
                   f"paper.runner.broker_fee_max_auto ({self.broker_fee_max_auto:g} USD) left for human review",
                   res)
'''.split("\n")
lines[idx[0]:idx[0]] = [(ind + m if m else "") for m in method]

# 3. the call, before the ledger reconciliation inside _reconcile_once
start = [i for i, l in enumerate(lines) if re.match(r"\s*def _reconcile_once\(self", l)][0]
end = next((i for i in range(start + 1, len(lines)) if re.match(r"\s*def ", lines[i])
            and len(lines[i]) - len(lines[i].lstrip()) <= len(lines[start]) - len(lines[start].lstrip())), len(lines))
idx = [i for i in range(start, end) if "Reconciler(" in lines[i]]
if len(idx) != 1:
    stop(f"expected one Reconciler( call inside _reconcile_once, found {len(idx)}")
ind = lines[idx[0]][: len(lines[idx[0]]) - len(lines[idx[0]].lstrip())]
lines.insert(idx[0], ind + "self._book_broker_fees()")

s = "\n".join(lines)
for need in ("BrokerError", "BOOK", "timedelta", "def event("):
    if need not in s:
        stop(f"runner.py has no {need}")
if not re.search(r"^from datetime import .*\bdate\b", s, re.M):
    stop("runner.py does not import date from datetime")
open(P, "w", encoding="utf-8").write(s)
try:
    py_compile.compile(P, doraise=True)
except py_compile.PyCompileError as exc:
    stop(f"runner.py does not compile: {exc}")
git("add", P)
print("runner.py merged (your version + the fee fix). Running tests, a few minutes...")
t = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-W", "ignore",
                    "tests/pipeline", "tests/execution"])
if t.returncode != 0:
    stop("tests failed (see above)")
c = git("commit", "--no-edit")
if c.returncode != 0:
    stop("git commit failed: " + c.stderr)
print("DONE: merge committed, all pipeline + execution tests passed.")
print(git("log", "--oneline", "-3").stdout)
