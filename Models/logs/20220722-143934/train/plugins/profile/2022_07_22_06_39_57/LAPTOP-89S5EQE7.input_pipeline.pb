	��?�@��?�@!��?�@	��*h<:@��*h<:@!��*h<:@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��?�@������?AV}��b�?Y�?�߾�?*�����܍@)      @=2F
Iterator::Modelo�ŏ1�?!��R�W@)��<,��?1���|��W@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��ͪ�Ֆ?!���<�@)�:pΈ�?10a��9N�?:Preprocessing2U
Iterator::Model::ParallelMapV2�+e�X�?!��_e�?)�+e�X�?1��_e�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate �o_Ή?!�9Q���?)�� �rh�?1����v�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�f��j+�?!-����@) �o_�y?1�9Q���?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4q?!X��-!�?)�J�4q?1X��-!�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!�gUW�u�?)	�^)�p?1�gUW�u�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap%u��?!n	��?)�J�4a?1X��-!�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 26.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2t32.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9��*h<:@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	������?������?!������?      ��!       "      ��!       *      ��!       2	V}��b�?V}��b�?!V}��b�?:      ��!       B      ��!       J	�?�߾�?�?�߾�?!�?�߾�?R      ��!       Z	�?�߾�?�?�߾�?!�?�߾�?JCPU_ONLYY��*h<:@b 