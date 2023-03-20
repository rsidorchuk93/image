The app takes user input as uploaded photo, runs age recognition model (VIT transformer model) and then displays image and estimated age group and confidence score

Deployment on AWS Beanstalk (required t2 medium server with 32 GB storage due to high pytorch memory requirement) and continuous deployment from GitHub via AWS CodePipeline
