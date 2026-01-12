# Setting Up Branch Protection

This guide explains how to configure branch protection rules to require tests to pass before merging PRs.

## Steps to Enable Required Status Checks

1. **Go to Repository Settings**
   - Navigate to your repository on GitHub
   - Click on **Settings** tab
   - Click on **Branches** in the left sidebar

2. **Add Branch Protection Rule**
   - Click **Add rule** or edit existing rule for `main`
   - In "Branch name pattern", enter: `main`

3. **Configure Protection Requirements**
   Check the following options:

   - ✅ **Require a pull request before merging**
     - Optional: Require approvals (1 or more reviewers)

   - ✅ **Require status checks to pass before merging**
     - Click to enable this option
     - Search for and select: **Test Summary**
     - Optional: Check "Require branches to be up to date before merging"

   - ✅ **Do not allow bypassing the above settings** (recommended)
     - This ensures even admins must pass tests

4. **Save Changes**
   - Scroll to bottom and click **Create** or **Save changes**

## What This Does

- PRs cannot be merged until the **Test Summary** check passes
- The Test Summary check requires ALL test matrix jobs to succeed:
  - Ubuntu + Python 3.11
  - Ubuntu + Python 3.12
  - Windows + Python 3.11
  - Windows + Python 3.12
- If any test fails, the PR will be blocked from merging
- A green checkmark will appear when all tests pass

## Testing the Protection

1. Create a test PR that breaks tests
2. Verify the "Merge" button is disabled
3. Fix the tests
4. Verify the "Merge" button becomes enabled once tests pass

## Troubleshooting

**"Test Summary" doesn't appear in status checks:**
- Make sure the workflow has run at least once on the main branch
- Push a commit to trigger the workflow
- Wait a few minutes and try again

**Tests are required but I need to merge anyway:**
- Admins can temporarily disable branch protection
- Or use "Administrator override" if configured
