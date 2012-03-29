This setup configures Jenkins to run on a single node.

Build Strategy
--------------

  * poll a Github repository
  * when a changed branch whose name matches the glob pattern `integrate-*` is found, that branch is merged to the local `master`
  * `master` is built with `scons` and unit tests are executed
  * If the build is successful:
    * If the integration branch exists on the remote, it is deleted from the remote
    * `master` is pushed back to the repository

The result is that the integration branch has been merged to `master` in the repository and no longer exists.

Configuration
-------------

1. Install Jenkins
2. Install the Jenkins plugins: 
  * SCons
  * GIT
  * Github
  * Python
3. Create a new build job
  * The name must not contain spaces
  * e.g. `thrust_unit_tests`
4. Add a helpful description, e.g.

    Polls thrust/thrust

    When a changed branch whose name matches integrate-* is found, that branch is merged to the local master branch

    master is built with scons and unit tests are executed

    If the build is successful:

    1. The integration branch is deleted on thrust/thrust
    2. master is pushed back to thrust/thrust

5. Fill in the box "GitHub project":
  * `http://github.com/thrust/thrust/`
6. Under **Source Code Management**
  1. Select **Git**
  2. Fill in the **Repository URL**
    * `git@github.com:thrust/thrust.git`
  3. Fill in the **Branch Specifier**
    * `origin/integrate-*`
7. Under **Build Triggers**
  1. Select **Poll SCM**
  2. Fill in the box **Schedule**: `* * * * *`
8. Under **Build**
  1. Select **Add build step** and choose **Invoke SCons script**
    1. Fill in **Invoke SCons script** boxes
      1. **Options**: `-j2`
      2. **Variables*: `host_backend=all device_backend=all arch=sm_20`
      3. **Targets**: `run_tests`
      4. **SConscript root directory**: `.`
      5. **SConscript file**: `SConstruct`
  2. Select **Add build step** and choose **Execute Python script**
    1. Fill in **Execute Python script** box with the following script

    # this script deletes the remote integration branch which triggered this build if it exists
    import sys
    import os
    import subprocess
    git_branch = os.environ['GIT_BRANCH']
    # poll the remote's list of branches
    remote_branches = subprocess.check_output(['git', 'branch', '-r'])
    if git_branch in set(remote_branches.split()):
      (remote, branch) = git_branch.split('/')
      print 'Deleting remote branch', branch
      command = ['git', 'push', remote, ':'+branch]
      subprocess.check_call(command)

9. Under **Post-build Actions**
  1. Check the box **Git Publisher**
    * Check the box **Push Only If Build Succeeds**
    * Check the box **Merge Results**
  2. Under **Branches**
    * Fill in the box **Branch to push**: `master`
    * Fill in the box **Target remote name**: `origin`

