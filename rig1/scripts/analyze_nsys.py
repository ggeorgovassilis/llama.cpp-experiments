import sqlite3, sys

db = sys.argv[1]
c = sqlite3.connect(db)
cur = c.cursor()

def q(sql, args=()):
    cur.execute(sql, args)
    return cur.fetchall()

def name(sid):
    r = q("SELECT value FROM StringIds WHERE id=?", (sid,))
    return r[0][0] if r else str(sid)

# decode phase = window spanned by mul_mat_vec_q kernels (1-token matvec, generation only)
b = q("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE shortName = (SELECT id FROM StringIds WHERE value='mul_mat_vec_q')")
decode_start, decode_end = b[0]
wall = decode_end - decode_start

print(f"decode phase: {decode_start/1e6:.1f} ms -> {decode_end/1e6:.1f} ms  (wall {wall/1e6:.1f} ms)")

print("\n== per-device DECODE busy (all kernels in decode window) ==")
rows = q("""
  SELECT deviceId, SUM(end-start), COUNT(*)
  FROM CUPTI_ACTIVITY_KIND_KERNEL
  WHERE start >= ? AND start <= ?
  GROUP BY deviceId ORDER BY deviceId
""", (decode_start, decode_end))
for dev, busy, n in rows:
    print(f"  GPU{dev}: busy={busy/1e6:8.1f} ms  kernels={n:6d}  util={busy/wall*100:5.1f}%")

print("\n== per-device DECODE matvec (mul_mat_vec_q) ==")
rows = q("""
  SELECT deviceId, SUM(end-start), COUNT(*), AVG(end-start)
  FROM CUPTI_ACTIVITY_KIND_KERNEL
  WHERE shortName = (SELECT id FROM StringIds WHERE value='mul_mat_vec_q')
  GROUP BY deviceId ORDER BY deviceId
""")
for dev, busy, n, avg in rows:
    print(f"  GPU{dev}: matvec={busy/1e6:8.1f} ms  n={n:5d}  avg={avg/1e6:.3f} ms")

print("\n== top kernels by DECODE busy ==")
rows = q("""
  SELECT shortName, SUM(end-start), COUNT(*)
  FROM CUPTI_ACTIVITY_KIND_KERNEL
  WHERE start >= ? AND start <= ?
  GROUP BY shortName ORDER BY 2 DESC LIMIT 10
""", (decode_start, decode_end))
for sid, busy, n in rows:
    print(f"  {name(sid):40s}: busy={busy/1e6:8.1f} ms  n={n:6d}")

print("\n== per-device DECODE memcpy ==")
rows = q("""
  SELECT deviceId, SUM(end-start), COUNT(*)
  FROM CUPTI_ACTIVITY_KIND_MEMCPY
  WHERE start >= ? AND start <= ?
  GROUP BY deviceId ORDER BY deviceId
""", (decode_start, decode_end))
for dev, busy, n in rows:
    print(f"  GPU{dev}: memcpy={busy/1e6:8.1f} ms  n={n:6d}")

print("\n== memcpy copyKind in decode window ==")
rows = q("""
  SELECT copyKind, COUNT(*), SUM(end-start)
  FROM CUPTI_ACTIVITY_KIND_MEMCPY
  WHERE start >= ? AND start <= ?
  GROUP BY copyKind ORDER BY 3 DESC
""", (decode_start, decode_end))
for ck, n, busy in rows:
    label = q("SELECT label FROM ENUM_CUDA_MEMCPY_OPER WHERE id=?", (ck,))
    lbl = label[0][0] if label else str(ck)
    print(f"  {lbl:28s}: n={n:6d}  busy={busy/1e6:8.1f} ms")

print("\n== synchronization in decode window ==")
rows = q("""
  SELECT syncType, COUNT(*), SUM(end-start)
  FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
  WHERE start >= ? AND start <= ?
  GROUP BY syncType ORDER BY 3 DESC
""", (decode_start, decode_end))
for st, n, busy in rows:
    label = q("SELECT label FROM ENUM_CUPTI_SYNC_TYPE WHERE id=?", (st,))
    lbl = label[0][0] if label else str(st)
    print(f"  {lbl:20s}: n={n:6d}  busy={busy/1e6:8.1f} ms")

print("\n== host-side API (sync + launch) in decode window ==")
rows = q("""
  SELECT nameId, COUNT(*), SUM(end-start)
  FROM OSRT_API
  WHERE start >= ? AND start <= ?
  GROUP BY nameId ORDER BY 3 DESC LIMIT 10
""", (decode_start, decode_end))
for nid, n, busy in rows:
    print(f"  {name(nid):45s}: n={n:6d}  busy={busy/1e6:8.1f} ms")
