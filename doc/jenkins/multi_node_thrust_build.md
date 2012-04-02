This setup configures Jenkins to run a distributed build across several nodes.

Build Strategy
--------------
  * poll a Github repository
  * when a changed branch whose name matches the glob pattern `integrate-*` is found, that branch is merged to the local `master`
  * `master` is built with `scons` in the combinatorial space of configurations and unit tests are executed
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
  * Join
3. Introduce build jobs
  1. Introduce a job to poll Github for changes
    1. Create a new build job
      * name it `poll_github_for_changes`
    2. Add a helpful description, e.g.
  
          ~~~
          Polls jaredhoberock/thrust-staging.git
  
          When a changed branch is found whose name matches integrate-*, several build jobs are forked
  
          If the build jobs are successful, they are joined by prune_branch_and_merge_changes
          ~~~

    3. Fill in the box "GitHub project": `http://github.com/jaredhoberock/thrust-staging`
    4. Under **Source Code Management**
      1. Select **Git**
      2. Fill in the **Repository URL**: `git@github.com/jaredhoberock/thrust-staging.git`
      3. Fill in the **Branch Specifier**: `origin/integrate-*`
    5. Under **Build Triggers**
      1. Select **Poll SCM**
      2. Fill in the box **Schedule**: `* * * * *`
    6. Under **Build**
      1. Select **Trigger/call builds on other projects**
        1. Fill in the box **Projects to build**: `test_and_integrate_branch`
        2. Check **Block until the triggered projects finish their builds*
        3. Select **Add Parameters**
          1. Choose **Predefined parameters**
            1. Fill in the **Parameters** box: `GIT_BRANCH=$GIT_BRANCH`
  2. Introduce a job to test and integrate an integration branch
    1. Create a new build job
      * name it `integrate_and_test_branch`
    2. Add a helpful description, e.g.
  
        Tests a specified Git branch. If the tests pass, the branch is integrated to master.
  
    3. Check **This build is parameterized**
      1. Select **Text Parameter**
        1. Fill in the **Name** box: `GIT_BRANCH`
        2. Fill in the **Description** box: `The git branch to test and integrate`
    4. Under **Source Code Management**
      1. Select **Git**
      2. Fill in the **Repository URL**: `git@github.com/jaredhoberock/thrust-staging.git`
      3. Fill in the **Branch Specifier**: `$GIT_BRANCH`
    5. Under **Post-build Actions**
      1. Check **Join Trigger**
        1. Check **Run post-build actions at join**
        2. Check **Trigger parameterized build on other projects**
          1. Fill in the box **Projects to build**: `prune_branch_and_merge_changes`
          2. Select **Add Parameters**
            1. Select **Current build parameters**
      2. Check **Trigger parameterized builds on other projects**
        1. Fill in the box **Projects to build**: `thrust_unit_tests_matrix`
        2. Select **Add Parameters**
          1. Select **Current build parameters**
  3. Introduce a job to build and execute unit tests
    1. Create a new build job
      * name it `thrust_unit_tests_matrix`
      * select multi-configuration job
    2. Add a helpful description, e.g.
  
          Builds and executes the Thrust unit tests over the entire combinatorial test space.
  
    3. Check **This build is parameterized**
      1. Select **Text Parameter**
        1. Fill in the box **Name**: `GIT_BRANCH`
        2. Fill in the box **Default Value**: `master`
        3. Fill in the box **Description**: `Git branch to test`
      2. Under **Source Code Management**
        1. Select **Git**
        2. Fill in the field **Repository URL**: `git@github.com/jaredhoberock/thrust-staging.git`
        3. Fill in the field **Branch Specifier**: `$GIT_BRANCH`
      3. Under **Configuration Matrix**
        1. Select **Add Axis**
          1. Select **User-defined Axis**
            1. Fill in the field **Name**: `HOST_BACKEND`
            2. Fill in the field **Values**: `cpp omp tbb`
        1. Select **Add Axis**
          1. Select **User-defined Axis**
            1. Fill in the field **Name**: `DEVICE_BACKEND`
            2. Fill in the field **Values**: `cuda omp tbb`
      4. Under **Build**
        1. Select **Add build step**
          1. Select **Execute Python script**
          2. Fill in the field **Script**:

          ~~~
          # the scons plugin doesn't work with multiconfig jobs
          # so launch scons manually
          # use python to make the launch process portable
          import subprocess
          import os
          host_backend=os.environ['HOST_BACKEND']
          device_backend=os.environ['DEVICE_BACKEND']
          targets = ['run_examples']
          command = ['scons', '-j2', 'host_backend='+host_backend, 'device_backend='+device_backend, 'arch=sm_20'] + targets
          subprocess.check_call(command)
          ~~~
  
  4. Introduce a job to delete the remote integration branch and merge its changes to `master`
    1. Create a new build job
      1. Name it `prune_branch_and_merge_changes`
    2. Add a helpful description, e.g.
  
          Deletes the specified branch from the remote repository and pushes merged changes to master.
  
    3. Check **This build is parameterized**
      1. Fill in the field **Name**: `GIT_BRANCH`
    4. Under **Source Code Management**
      1. Select **Git**
        1. Fill in the field **Repository URL**: `git@github.com:jaredhoberock/thrust-staging.git`
        2. Fill in the field **Branch Specifier**: `$GIT_BRANCH`
    5. Under **Build**
      1. Select **Add build step**
      2. Select **Execute Python script**
 1. Fill in the field **Script**:
 
        ~~~
        # this script deletes the remote branch specified by the parameter GIT_BRANCH
        # much of the string manipulation here results from the fact that GIT_BRANCH
        # may be either "repo/branch" or just "branch"
        import os
        import re
        import subprocess
        git_branch = os.environ['GIT_BRANCH']
        # ignore branches which aren't temporary integration branches
        if re.match('(.*/)?integrate-.*', git_branch):
          # prepend 'origin/' if it doesn't exist
          if 'origin/' not in git_branch:
            git_branch = 'origin/' + git_branch
          # poll the remote's list of branches
          remote_branches = subprocess.check_output(['git', 'branch', '-r'])
          if git_branch in set(remote_branches.split()):
            (remote, branch) = git_branch.split('/')
            print 'Deleting remote branch', branch
            command = ['git', 'push', remote, ':'+branch]
            subprocess.check_call(command)
          else:
            print 'Ignoring non-existing remote branch', git_branch
        else:
          print 'Ignoring non-integration branch', git_branch
        ~~~

    6. Under **Post-build Actions**
      1. Check **Git Publisher**
        1. Check **Push Only If Build Succeeds**
        2. Check **Merge Results**
        3. Under **Branches**
          1. Fill in the field **Branches to push**: `master`
          2. Fill in the field **Target remote name**: `origin`
4. Introduce worker nodes 
  1. TODO

