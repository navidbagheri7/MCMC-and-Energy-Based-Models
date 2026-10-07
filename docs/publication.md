# Publish the locally prepared repository

The owner publishes the reviewed local release. No GitHub push, remote or site
deployment is created automatically. Code, procedural outputs and owned energy
weights use MIT; public real-data derivatives retain their original attribution.
Original coursework and private environments remain excluded.

Create an empty repository under your account, then review the release checklist:

```bash
git remote add origin git@github.com:YOUR_ACCOUNT/sampling-energy-lab.git
git push -u origin main
git push origin v1.0.0
```

The public explorer already works locally. To host it, choose GitHub repository
**Settings → Pages → Deploy from a branch → main → / (root)**. The supplied
`.nojekyll` keeps raw metric and asset paths unchanged. The demo URL becomes
`https://YOUR_ACCOUNT.github.io/YOUR_REPOSITORY/docs/explorer.html`. Keep both
`docs/` and `assets/` in the published tree. Add the URL to the repository's
About/Website field and README after deployment. The README images are real
browser previews; the HTML supplies the actual controls.
