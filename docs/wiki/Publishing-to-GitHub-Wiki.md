# Publishing to GitHub Wiki

This folder (`docs/wiki/`) is formatted as a GitHub Wiki content bundle.

## 1) Ensure Wiki is enabled

In the GitHub repository settings, enable **Wiki**.

If wiki is disabled, cloning/pushing `https://github.com/<owner>/<repo>.wiki.git` will fail with `repository not found`.

## 2) Sync local bundle to wiki repo

```bash
# choose owner/repo
OWNER=ACS-USP
REPO=replicant

rm -rf /tmp/${REPO}.wiki

git clone https://github.com/${OWNER}/${REPO}.wiki.git /tmp/${REPO}.wiki
rsync -av --delete --exclude .git docs/wiki/ /tmp/${REPO}.wiki/

cd /tmp/${REPO}.wiki
git add .
git commit -m "docs: update project wiki"
git push origin master
```

Note: GitHub Wiki default branch is commonly `master`.

## 3) Verify

Open:

`https://github.com/ACS-USP/replicant/wiki`

Check:

1. Sidebar navigation appears
2. Mermaid diagrams render
3. Internal page links resolve
