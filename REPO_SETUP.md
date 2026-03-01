# Using this as a separate repository

This directory is a clean export of the refactored entity attention analysis code, with no experiment logs, job outputs, or secrets.

To publish as a new GitHub repository:

1. Copy this folder to the location you want (or rename the parent):
   ```bash
   cp -r entity_attn_analysis_public /path/to/your/new-repo-name
   cd /path/to/your/new-repo-name
   ```

2. Initialize git and make the first commit:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: entity attention analysis (refactored for public release)"
   ```

3. Create a new repository on GitHub, then:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

Then add your paper citation and license to `README.md` as needed.
