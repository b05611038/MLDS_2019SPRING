import sys
import tensorflow as tf 

from cs_module import correlation_score
from lm_module import Language_model, test, Data_loader

def main(input_file, output_file):
    data_loader = Data_loader(output_file, 400, 30)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = Language_model()
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('model'))
        perplexity = test(sess, model, data_loader)
        print ('perplexity      : {0:3f} (baseline: < 100)'.format(perplexity))
    t = correlation_score(input_file, output_file)
    correlation = t.predict()
    print ('correlation score : {0:.5f} (baseline: > 0.45)'.format(correlation))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
