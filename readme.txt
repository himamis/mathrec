###############
### Install ###
###############
Download the Google Cloud SDK from https://cloud.google.com/sdk/ and install it. Now you 
can use gcloud and gsutil. Insure that you initialized by using 'gcloud init'. When doing 
the initialization use 'Image2Latex' as your project and something like 'uscentral1' as 
compute zone. How the initialization exactly works you also find on the link above. 

###################
### Upload data ###
###################
For uploading an huge amount of data to Google Cloud Storage, I would advise against doing 
that by your browser. Instead, it is better to use 'gsutil cp ...' for uploading it. 
If the data is very big, so that the 'gsutil cp ...' command lasts several hours to days, 
I would recommend you to use the MAC Server of GeoGebra. Here a scenario:
	1. You have a Java programm called render.jar that takes two arguments -inp and -outp
	   on your computer. -inp is path to LaTeX formulas file and -outp is output directory, 
	   where the rendered formulas get saved.
	2. Upload render.jar to the MAC server by enter in your terminal:
		'pscp /path/to/file/render.jar geogebra@140.78.116.132:/path/to/dest/ -pw J.K.U.54'
	   Maybe you have to use the above command without pw option and enter the pw afterwards.
	   Also upload the formulas file to the server with the same way.
	3. Login to the MAC Server using putty by enter in your terminal:
		'putty geogebra@140.78.116.132 -pw J.K.U.54'
	4. Enter 'cd /path/to/dest/'
	5. Enter 'nohup java -jar render.jar -inp=formulas.lst -outp=./images/ &'
	   When using 'nohup ... &' you can close your putty connection without stopping 
	   the execution of the jar file. This is useful if the jar file's execution lasts 
	   for several hours or days.
	6. To verify the progress of the java program, you can do following:
		Change to images folder by entering 'cd /path/to/dest/images'.
		Find out count of files in this directory by entering 'ls -1 | wc -l'.
	7. If the java process has finished you can upload it to Google Cloud Storage by using 
		'nohup gsutil -m cp -r ./ gs://image2latex-mlengine/path/to/dest/ &'
	   The -m option forces gsutil to use multi threads and processes. The -r option stands 
	   for recursive and is used to upload directories. gcloud and gsutil are already installed 
	   on the MAC Server and gcloud is also initialized for the Image2Latex project I think. 
	8. To verify the progress of the upload you can enter:
		'gsutil du gs://image2latex-mlengine/path/to/dest | wc -l'
	   to get the count of already uploaded files.
	9. No number 9 :))

############################
### Start a training job ###
############################
To start a training job just copy the text in start_job_linux.txt, respective start_job_windows.txt 
and paste it into a terminal window. Keep in mind that gcloud has to be in your path variable. 
The command in the start_job_windows.txt has an little bug that occurs when you start a job betwenn 
00:00 and 09:59. The reason is that the jobname includes the date and the time to get different job 
names for each job. But if the time is between the previouse mentioned time, windows returns for the 
hours _x, where _ stands for a space and x for the hour. So you have a space in your job name and that 
is not allowed. A possible workaround is to replace %time:~0,2% with %time:~1,1%.
If you have started your job, you can verify its status at:
'https://console.cloud.google.com/mlengine/jobs?project=image2latex&authuser=1&organizationId=1061698309396&pli=1'
Keep in mind that it is not unsual that a job ends after 55 to 60 secondes. If you look at the logs, 
you might read something with resources. If this happens just start the job again.

###################
### Other Infos ###
###################
When you want to start the training locally, just don't add the --gcs argument when executing a 
python file. To preprocess data use preprocess.py. For training use task.py and for predicting 
images use predict.py. 
Keras 2.0.4, tensorflow 1.4 and python 2.7 are used on the google cloud machine.
If you want to execute for example task.py locally but you take the data from Google Cloud Storage,
you have to use the --gcs argument. See /trainer/__init__.py for more information.
Also the apache_beam implementation gcsio only works for python 2.7. 