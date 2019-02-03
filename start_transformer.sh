if [ -z "$1" ] ; then
    echo "First argument must be the name"
    exit 1
fi

if [ -z "$2" ] ; then
    echo "The second argument must be the project index"
    exit 2
fi

if [ $2 == "3" ] ; then
    PROJECT_ID="nomadic-raceway-224107"
    BUCKET_NAME="image2latex3"
elif [ $2 == "4" ] ; then
    PROJECT_ID="pivotal-store-224110"
    BUCKET_NAME="image2latex3"
elif [ $2 == "2" ] ; then
    PROJECT_ID="true-eye-223916"
    BUCKET_NAME="image2latex3"
elif [ $2 == "1" ] ; then
    PROJECT_ID="image2latex"
    BUCKET_NAME="image2latex-mlengine"
else
    echo "Unknown project index"
    exit 1
fi


NAME=$1

echo "Name is: "$NAME
echo "Project id is: "$PROJECT_ID
echo "Bucket name is: "$BUCKET_NAME



DATE=`date '+%Y_%m_%d__%H_%M'`
JOB_NAME="im2lx_"$DATE$NAME
CONTAINER="gs://"$BUCKET_NAME
OUTPUT_PATH=$CONTAINER"/output_transformer"
TB_PATH=$CONTAINER"/logdir3"
PROF_PATH=$CONTAINER"/profiling"

#REGION="us-central1"
#REGION="europe-west4"
#REGION="us-east1"
REGION="us-west1"
DATA_URL="token_trace/"
OUTPUT_URL="output"

echo "Job: "$JOB_NAME" running on: "$OUTPUT_PATH

gcloud config set project $PROJECT_ID
gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.12\
    --module-name transformer.trainer \
    --package-path transformer/ \
    --region $REGION \
    --config config.yaml \
    -- \
    --gcs $BUCKET_NAME \
    --data-base-dir $DATA_URL \
    --model-dir $OUTPUT_URL \
    --start-epoch -1 \
    --git-hexsha `git rev-parse HEAD` \
    --gpu 0 \
    --tb $TB_PATH \
    --tbn $JOB_NAME \
    --allow-soft-placement t \
    --epv 20
#    --verbose-summary t

echo "To look on the results in tensorboard, go to http://localhost:800"$2
echo "If tensorboard is not started, call \`tensorboard --logdir "$TB_PATH" --port 800"$2"\`"
