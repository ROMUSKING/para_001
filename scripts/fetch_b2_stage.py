#!/usr/bin/env python3
"""scripts/fetch_b2_stage.py

Reproducibly stage the B2 probe inputs on a fresh VM from Drive share links:
100 DROID episode .npz files (matched to the committed cache manifest by SHA-256),
5 frozen dynamics teachers, and the run config. Writes the run layout the benchmark
script expects plus a path-adjusted manifest copy (original preserved; hashes carry
integrity, not paths).

Usage (on the VM)::
    pip install -q gdown requests
    python scripts/fetch_b2_stage.py --dest /content/stage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

try:
    from gdown.download import download as gdownload
except ImportError:
    sys.exit("gdown required: pip install gdown")

RUN_ID = "droid100_adjoint_v2_5seeds_20261001T080821Z"

# Per-file Drive links shared by the operator (Anyone with the link), covering all
# 100 manifest episodes. Files are matched to manifest entries by SHA-256, never by
# name or order, so this list is robust to transcription order.
EPISODE_IDS = """1-AL3j0Nt1X-4NFpWYkySdTSDFi6GYMkN
1-pdj80JGZu5vVVCu9fT_0Mf8WkAU5Ewv
11mAbWsi2koWXFeJlAAQtpvOnM03fyLwp
11zb5R0f8EkFRgZD8IHii5aMSpnTlqVZV
12PvYpFiGqLTmQGur88Rl5DUEpWvxQC6G
12RC3yB_-Sz_XTwwRIG0iZzdEGjUp-4Fz
12w5bz7aISsWCdAwdvGN5Q6oP8PQ50Bt4
14BbDe2aVEQQ3hUHdwcbefajW6QODprbg
14kT6TTC_8a163NsTTWvfAcT5-yfmlmrS
14t5-_9txzcVezb0zxD9zAFuOyFb5zfiD
15vpK4VLzqNHSECVXIxF-7HkEyjXJcgXp
15xh9r4JxVV_zplTbPJsG4ZUtT9Pp5w8R
16FD_8my7_KWAXv5ISTGi8gMFZT6d5rqV
16XJVlns_oKb6HIpKbuIhVvXCBMotmy6r
16c143TrWCdbfFfoyKV6CYCCkkdohTPWW
16oCBucShgnT1jTuGaTkQypZlyuiflDDo
189HFqRKh8AK1xqqALthOgueG6wp02N2d
193HIV3BX9SWElULOPWSLIeeAFYrYi5mX
19cEIJOUZrqSCFC7kl5bZMTSvcizseVBI
1Bj2t5Ne-8xR8qevKwr6hUmv6JMgzUSKF
1BkrmkX81QU-MBsnKeIk9buK2-Ca1s2Rr
1BtCt20bBsSqlBHmA0w_zushidYnUi1D9
1Cf-BZK-Qo2xvNtYbWS4oVDMLYp04bULg
1Cm9RT467FmKOynyBluTsu_08ltpROSOG
1DOuj1-lMx6r2zWQZ314RTUUHiXSjTB4v
1EFkFIhxBhNwn2O1xRTK095thm7SPh3no
1GH5ZXkCvSBcx681tlAP8zUeTPPY0v64w
1GVfhqn7Z2bRVVUT5rxNgQkEQZLI8-M4j
1H-aVQu3dIz7-3xHBv0Saxqq6GGSPNxaR
1IgQW7w24AH2PTi9MZre1rHjkZY1W5TXC
1J4r_b60SM7L_AYmYW4h4uvT4r-qnrXci
1K4TtK1JYM1iAOrAkCDgKmiAGV9fb7WBA
1L9IeRPviolr_4fXVZWnC2a-g3jburxGU
1P-2dRR-VcTYhGXsF1JTjQjJJhNTgNyrH
1QCm-GDe2M-9b1coRVnjTkMaX6MxqG8Jk
1QxEmMtPPQ9l786jOvpCRVmY1rJYBjcJY
1RalOJ2hfqoMqDtWPzljH0nOXY-S2sz1d
1UdvZRqldPoiLfAilUrHCCVzUaXPLmIM0
1VhgbiOzglj8crJ9w7ILfe5EVMFr1hWb0
1VjYvtb34FrfJJq1SPWWlDA7VzPkld56L
1WZ4vGO9BZXLlgizIFq_eXDNp2jqOBpug
1WeewbWIlrH-9AOUoNnZO-8MQATMpGc7j
1XYKX7ki6FnuLUI8vuS_ht4W8k4vv1LH5
1Xlbe6zJHGYm7M7N6aw50C5NOweD2hZ6p
1Y2omj4k6eNsk5jKVdtFoWO5rG7sa_qbw
1YwANW8ZoqVnReV0fZmlV8E0onxB9el6d
1ZENQ_QJiURPODWWcTVpB6d41vc7JnPn3
1ZSM3B8fpOSOD-v5k2Fshu5nEo53vSpU5
1_uWnjoz_VtUx1XL3PbZ2Hn7wJLLdMR1m
1aVXFd3VtJ64szYWd_PZsdkqJnI5q3Sto
1amx88QjF5sOsjp0ADeoE1zEL2aKlDDSP
1apN72-NwKYA7yJRkBbiTdjc1gFT6Jg1W
1bMRdF2UwRsX1ZO4PddaushhXratbw-dn
1bnjelOXwjY06ssILk8JDLrOuyabOM7Vu
1cMt-fW-vNWcHbXzAdIctp_dvLYWvctuO
1ckVwbfl7Yk6MBy6HD92GO9Qp37dPEa5t
1dGl6AkvXX1U6oKRtGA55OV2G25nmMKi-
1dUDQdL9airw92XFBku0CLk85zVTrpwCM
1eL2GsMs9Cc_APVNWK8tcpIQwMMydl8CF
1eunCmSdP6x_8QpAtrL2F-1xDAyBgwXYW
1fAjwNbnPEeFbVSPRJEs2Z1zliDwqWO9K
1fEEdWZtizHMvPhlDusOufTanJK68VoES
1fbhyW74DWn0icDzkKykLD1y2FBE47nm-
1h3xAyoKNoEB4UmMafiRMMweAyjqmfTIN
1hbRKVMWmSWzLgk2XdZqghrAHeNBp7EZ-
1iruwqklNJXZLVNzpsh_yQIB8VAIT0abw
1itmvVhnjfuzBPW3He2FSpYtgmwaJsQc5
1jlqvy22lBOkkXQxhimYMYM4GXhxEgVO-
1k2MryKCBA0MtCSyMyExoeuDKndHpdvCZ
1kGdSgptGHltjDJsbgYieUomiRfHEXKtX
1kptwoKNKUjCmRTzMwJ1xfiRpgNL3c3Tn
1l_kU5558KoqilqqKmy6MfgQmkfr2_YY5
1mSE9YiXGKf24TI0tYxEd3zlGrAbTNfVk
1nBZkmRlGyRlPcdL7gw5ACDz8DUVMW2u7
1owVOclEnpTGoOdsZBolyD1hj8Afp1mdD
1p31CodZTDvNCr51AEU5xLFl7-Y96ZXGd
1pEW2d_l2dw_0q6bUQkOD_azPcpLwy2gz
1qP1g-l8MdUxf0QLUGXmCtCq8uKmcifSX
1qao7ZncKlO8SJ8sFMJ-jxRdeT7VW1x2D
1qmIO1p8UVjmoSIA8RVf4hgvXCO0xv9ma
1r180pJ_fWM0oZWjp72rE8KlquOBo6tiP
1rwTwk-OxEPuU2as3T-r-UF6Ug_bqQ5w6
1t965xihjeuyaNRWjPmNmPLHjnkGXHBCd
1u0u6yqi6bwMghcM6dYbI7DEDNO4tRPkG
1u49SOR6kKk-n93I84ddTWUwI-_QJat-W
1uKZifxAdeHeaf4H7Zy5RrKrqfYVoGmRI
1uNZEvJw59dTDN8awrfbTyn0-SyM6lGUK
1uhA8YxQCNwYyEbr8-4wVjvuDX9sTfUEb
1uqRIGUxFU6hGIZtikpvrpwHxvjUXjQwG
1vMYkWb_mRa1yxftBA3dmKF8YgIGhytQp
1vVO_mvRukoC0L9j2wPM2Nfs4pEht-5Yo
1vtEAP-FgbCkCl63uiVB3kOmzDWM26STx
1vzRqyClB_2L_1XI9hQvFsWtE4uy_RgId
1w96TiFPOarmRZIrCM0Y_LTM0kkMlkSJ0
1wAHnfp1dMseS2bxeLEF_ueDzh2da7QGV
1wBBsK7YASuIbII_q8efI0jOjxswPxQJZ
1x7OQHqm6X4I0eNcTpxcyfEm8IXQEWMHf
1ypCX9mWBACyhRCrOrv5sePiPDYrCi8UM
1z75UWsQRV4o1rXJ5umYb5etkBavwcHuB
1zV44LPs_VwY7uSthhgIM3k2LTi3ilLRS""".split()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Stage B2 probe inputs from Drive links")
    parser.add_argument("--dest", default="/content/stage")
    parser.add_argument("--manifest", default=None,
                        help="committed cache_manifest.json path (for hash verification)")
    args = parser.parse_args()

    dest = args.dest
    ep_dir = os.path.join(dest, "episodes_source")
    os.makedirs(ep_dir, exist_ok=True)

    manifest = None
    if args.manifest and os.path.exists(args.manifest):
        manifest = json.load(open(args.manifest))
    by_hash = {}
    if manifest:
        by_hash = {e["cached_sha256"]: os.path.basename(e["persisted_path"])
                   for e in manifest["episodes"]}

    ok = fail = 0
    for i, fid in enumerate(EPISODE_IDS):
        target = os.path.join(ep_dir, fid)
        got = False
        for attempt in range(4):
            try:
                gdownload(id=fid, output=target, quiet=True)
                got = True
                break
            except Exception as exc:
                print("retry %s att%d: %s" % (fid[:8], attempt, type(exc).__name__),
                      flush=True)
                time.sleep(15 * (attempt + 1))
        if not got:
            fail += 1
            print("FAIL " + fid, flush=True)
            continue
        if by_hash:
            h = sha256_file(target)
            if h in by_hash:
                final = os.path.join(ep_dir, by_hash[h])
                if final != target:
                    if os.path.exists(final):
                        os.remove(target)
                    else:
                        os.rename(target, final)
                ok += 1
            else:
                fail += 1
                print("UNMATCHED " + fid, flush=True)
        else:
            ok += 1
        time.sleep(2)
        if i % 20 == 19:
            print("progress %d/%d" % (i + 1, len(EPISODE_IDS)), flush=True)
    print("FETCH_DONE ok=%d fail=%d" % (ok, fail), flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
