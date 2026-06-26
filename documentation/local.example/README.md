# Site-specific documentation templates

Tracked **templates** with placeholders. Copy into `documentation/local/` (gitignored) and replace placeholders with real facility values.

```bash
mkdir -p documentation/local
cp site-procServ.md documentation/local/my-beamline-procServ.md
# edit documentation/local/my-beamline-procServ.md
```

Do **not** put real hostnames, internal proxy URLs, or facility filesystem paths in files under `documentation/` that are committed to the public repo.

Security policy and site naming conventions belong in **`documentation/local/SECURITY.md`** (gitignored), a facility wiki, or a private repository — not in the public GitHub tree.

Run before pushing:

```bash
cp documentation/local.example/check-patterns.txt.example documentation/local/check-patterns.txt
# edit check-patterns.txt for your facility
./scripts/check-public-docs.sh
```

## Templates

| File | Purpose |
|------|---------|
| [site-procServ.md](site-procServ.md) | procServ, paths, PREFIX, Pixi on beamline |
