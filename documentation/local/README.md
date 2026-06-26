# Site-specific documentation (not in public git)

This directory is **gitignored** except for this README. Store facility material here:

- **`SECURITY.md`** — what must not be pushed to public git; pre-push checklist
- Runbooks (hosts, proxy, procServ, paths, PV prefixes)
- **`check-patterns.txt`** — extra regexes for `scripts/check-public-docs.sh`

Longer term, mirror or move this content to a facility **wiki** or **private** repository.

## Setup

```bash
cp -r documentation/local.example/* documentation/local/
cp documentation/local.example/check-patterns.txt.example documentation/local/check-patterns.txt
# Add documentation/local/SECURITY.md and runbooks; edit check-patterns.txt
```

## See also

- [../local.example/README.md](../local.example/README.md) — tracked templates (no security policy in public repo)
