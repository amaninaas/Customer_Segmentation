		�c�@	�c�@!	�c�@	�C�Z�8@�C�Z�8@!�C�Z�8@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$	�c�@46<�@A�/�$�?Yz�,C��?*	fffff�@2F
Iterator::Model��#����?!��B���W@)�k	��g�?1\�����W@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate䃞ͪϕ?!u6Jܷ)�?)"��u���?13�@���?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��_�L�?!yr��y��?)�J�4�?1�R3Ή6�?:Preprocessing2U
Iterator::Model::ParallelMapV2vq�-�?!�����?)vq�-�?1�����?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip&S���?!����'@)� �	�?1a��\�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!�% %��?)	�^)�p?1�% %��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mbp?!p2�'�?)����Mbp?1p2�'�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapg��j+��?!Ǡ���?)�J�4a?1�R3Ή6�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 24.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2t50.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�C�Z�8@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	46<�@46<�@!46<�@      ��!       "      ��!       *      ��!       2	�/�$�?�/�$�?!�/�$�?:      ��!       B      ��!       J	z�,C��?z�,C��?!z�,C��?R      ��!       Z	z�,C��?z�,C��?!z�,C��?JCPU_ONLYY�C�Z�8@b 