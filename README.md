# Waldo Finder App

Mobile app to help finding waldo using Tensorflow lite. 

## TODO
- [ ] Implement Android app
- [ ] Implement iOS app
- [ ] Improve detection using different tflite model

## Find the data

#### Images
Some folks already did the hard part of getting the images and the most important to find waldo in all of them: https://github.com/vc1492a/Hey-Waldo.

I decided to modify the images to increase the sample size and reduce images size.

#### Annotations
To generate annotations I used [LabelImg](https://github.com/vc1492a/Hey-Waldo.), which will save them as XML in PASCAL VOC format.

## Prepare the data

#### Initialization
```
export WALDO_REPO=[YOUR_FOLDER]
```

#### Generate CSV file from XML annotations

```
python xml_to_csv.py
```

#### Split the data

```
python split_labels.py
```

#### Generate TF Records
```
python generate_tfrecord.py --label_map_path=training/object-detection.pbtxt --csv_input=data/train_labels.csv --image_dir=images --output_path=data/train.record

python generate_tfrecord.py --label_map_path=training/object-detection.pbtxt --csv_input=data/test_labels.csv --image_dir=images --output_path=data/test.record
```

## Retrain models
### List of trained models
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models

### Add tensorflow & models repo
```
git submodule update init

export PYTHONPATH=$PYTHONPATH:$WALDO_REPO/tensorflow_models/research:$WALDO_REPO/tensorflow_models/research/slim

cd tensorflow_models/research
protoc object_detection/protos/*.proto --python_out=.
cd ../..
```

## Retrain SSD MobileNet v2
```
export MODEL_DIR=${WALDO_REPO}/models/ssd_mobilenet_v2
export PIPELINE_CONFIG_PATH=${MODEL_DIR}/pipeline.config
export NUM_TRAIN_STEPS=10
export NUM_EVAL_STEPS=1
python tensorflow_models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --num_eval_steps=${NUM_EVAL_STEPS} \
    --logtostderr
```

##### Optional retrain on Google Cloud platform
To speed up training, yu can use the Google Cloud platform. There is a very good [tutorial](https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193?linkId=54246631) to train model using Cloud TPUs. 

###### Upload data
```
gsutil -m cp -r data/*.record gs://${YOUR_GCS_BUCKET}/data/
gsutil -m cp -r data/*.pbtxt gs://${YOUR_GCS_BUCKET}/data/
```


###### Upload SSD MobileNet v2 model
```
gsutil -m cp -r models/ssd_mobilenet_v2/* gs://${YOUR_GCS_BUCKET}/data/
```

###### Start training
```
gcloud ml-engine jobs submit training `whoami`_object_detection_`date +%s` \
--job-dir=gs://${YOUR_GCS_BUCKET}/train \
--packages tensorflow_models/research/dist/object_detection-0.1.tar.gz,tensorflow_models/research/slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
--module-name object_detection.model_main \
--runtime-version 1.8 \
--scale-tier BASIC_GPU \
--region us-central1 \
-- \
--model_dir=gs://${YOUR_GCS_BUCKET}/train \
--tpu_zone us-central1 \
--pipeline_config_path=gs://${YOUR_GCS_BUCKET}/data/pipeline.config
```


```
gcloud ml-engine jobs submit training `whoami`_object_detection_eval_validation_`date +%s` \
--job-dir=gs://${YOUR_GCS_BUCKET}/train \
--packages tensorflow_models/research/dist/object_detection-0.1.tar.gz,tensorflow_models/research/slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
--module-name object_detection.model_main \
--runtime-version 1.8 \
--scale-tier BASIC_GPU \
--region us-central1 \
-- \
--model_dir=gs://${YOUR_GCS_BUCKET}/train \
--pipeline_config_path=gs://${YOUR_GCS_BUCKET}/data/pipeline.config \
--checkpoint_dir=gs://${YOUR_GCS_BUCKET}/train
```

###### Tensorboard
```
tensorboard --logdir=gs://${YOUR_GCS_BUCKET}/train
```


###### Generate frozen graph

```
export CONFIG_FILE=${WALDO_REPO}/models/ssd_mobilenet_v2/train/pipeline.config
export CHECKPOINT_PATH=${WALDO_REPO}/models/ssd_mobilenet_v2/train/model.ckpt-9634
export OUTPUT_DIR=/tmp/

# Export graph for tensorflow lite
python ${WALDO_REPO}/tensorflow_models/research/object_detection/export_tflite_ssd_graph.py \
--pipeline_config_path=$CONFIG_FILE \
--trained_checkpoint_prefix=$CHECKPOINT_PATH \
--output_directory=$OUTPUT_DIR \
--add_postprocessing_op=true

# Optimize graph for tensorflow lite using TOCO
cd ${WALDO_REPO}/tensorflow | \
bazel run -c opt tensorflow/contrib/lite/toco:toco -- \
--input_file=$OUTPUT_DIR/tflite_graph.pb \
--output_file=$OUTPUT_DIR/detect.tflite \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
--inference_type=FLOAT \
--mean_values=128 \
--std_values=128 \
--change_concat_input_ranges=false \
--allow_custom_ops

```

##### Test retrained models
```
python test_model.py ./models/ssd_mobilenet_v2/train/frozen_inference_graph.pb images/1_1.jpg
```

## Android app
#### prepare model
```
cp ${OUTPUT_DIR}/detect.tflite ${WALDO_REPO}/tensorflow/tensorflow/contrib/lite/examples/android/app/src/main/assets/

echo "???\nwaldo" > ${WALDO_REPO}/tensorflow/tensorflow/contrib/lite/examples/android/app/src/main/assets/waldo.txt
```

##### Change the config to use our retrained model and our label file
```
sed -i -e 's#@tflite_mobilenet_ssd_quant//:detect.tflite#//tensorflow/contrib/lite/examples/android/app/src/main/assets:detect.tflite#g' ${WALDO_REPO}/tensorflow/tensorflow/contrib/lite/examples/android/BUILD

sed -i -e 's#coco_labels_list.txt#waldo.txt#g' ${WALDO_REPO}/tensorflow/tensorflow/contrib/lite/examples/android/app/src/main/java/org/tensorflow/demo/DetectorActivity.java

sed -i -e 's#TF_OD_API_IS_QUANTIZED = true#TF_OD_API_IS_QUANTIZED = false#g' ${WALDO_REPO}/tensorflow/tensorflow/contrib/lite/examples/android/app/src/main/java/org/tensorflow/demo/DetectorActivity.java

sed -i -e 's#TFL Detect#Waldo Detector#g' ${WALDO_REPO}/tensorflow/tensorflow/contrib/lite/examples/android/app/src/main/res/values/base-strings.xml
```

#### Build & install app

```
cd ${WALDO_REPO}/tensorflow
```

###### Configure the SDK and NDK path. You need to update path with what's confgured on your machine.
```
echo '
android_sdk_repository(
    name = "androidsdk",
    path = "PATH_TO_SDK",
)

android_ndk_repository(
    name = "androidndk",
    path ="PATH_TO_NDK",
)
' >> WORKSPACE
```


###### Build the Android app
```
bazel build -c opt --cxxopt='--std=c++11' //tensorflow/contrib/lite/examples/android:tflite_demo
```

###### Install the Android app
```
adb install -r -f bazel-bin/tensorflow/contrib/lite/examples/android/tflite_demo.apk
```

