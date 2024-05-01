#!/bin/sh

git fetch --all

gh pr list --state open --label $1 --limit 500 >branches.list
#git branch -r >branches.list

#cat branches.list
cat branches.list|grep diff\/ |sed 's|^.*diff/|diff/|g' |sed 's|\t.*||g' >branches2.list
#cat branches.list|grep \/diff\/ |sed 's|^.*(diff/.*?)\s/.*$|$1|g' >branches2.list
cat branches2.list                

#git config --global user.email ${{ github.actor }}@users.noreply.github.com
#git config --global user.name "${{ github.actor }}"

# Fetch a list of open pull requests using gh pr list and extract the numbers
pr_numbers=$(gh pr list --state open --json number | jq -r '.[].number')

# Loop through the list of pull request numbers
for pr_number in $pr_numbers; do
    echo "Pull Request Number: $pr_number"
    gh pr edit $pr_number --auto-merge --merge --squash
done

while read branch; do
  echo "Branch: $branch"
  git checkout $branch
  git pull origin $branch
  git merge main
  echo "git commit --allow-empty -m 'Triggering Autotest Workflow Re-Run'"
  git commit --allow-empty -m 'Triggering Autotest Workflow Re-Run'
  echo "git push origin $branch"
  git push origin $branch
done <branches2.list  
