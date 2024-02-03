# Event Processing



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.nist.gov/gitlab/ncnrdata/event-processing.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.nist.gov/gitlab/ncnrdata/event-processing/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

## Name
NCNR event mode data

## Description
Tools for processing and visualizing event streams from NCNR instruments

## Installation

Use pip installation for end user tools
```sh
pip install https://gitlab.nist.gov/gitlab/ncnrdata/event-processing.git
```

## Usage

To run the webservice and gui
```sh
uvicorn event_processing.rebinning_api.server:app
python -m event_processing.rebinning_client.demo
```

## Contributing

Source lives in the NIST gitlab repository. Clone using:
```sh
git clone git@gitlab.nist.gov:gitlab/ncnrdata/event-processing.git
pip install -e event-processing
```
To run some basic tests:
```sh
python -m event_processing.rebinning_api.server check
python -m event_processing.rebinning_api.client
```

You will sometimes want to clear out the cache during development:
```sh
python -m event_processing.rebinning_api.server clear
```
Generally this happens automatically when you bump server.CACHE_VERSION,
but you may want to trigger it manually if you are playing with code timing.

## Authors and acknowledgment
Paul Kienzle, Brian Maranville

## License
This code is a work of the United States government and is in the public domain.

## Project status
On going.