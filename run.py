import argparse, time, os, operator

import tensorflow as tf
import source.connector as con
import source.tf_process as tfp
import source.datamanager as dman

def main():

    os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    dataset = dman.Dataset()

    agent = con.connect(nn=FLAGS.nn).Agent(\
        dim_h = dataset.height, \
        dim_w = dataset.width, \
        dim_c = dataset.channel, \
        num_class = dataset.num_class, \
        ksize = FLAGS.ksize, \
        learning_rate = FLAGS.lr, \
        path_ckpt = 'Checkpoint')

    time_tr = time.time()
    tfp.training(agent=agent, dataset=dataset, \
        batch_size=FLAGS.batch, epochs=FLAGS.epochs)
    time_te = time.time()
    tfp.test(agent=agent, dataset=dataset, batch_size=FLAGS.batch)
    time_fin = time.time()

    print("Time (TR): %.5f [sec]" %(time_te - time_tr))
    te_time = time_fin - time_te
    print("Time (TE): %.5f (%.5f [sec/sample])" %(te_time, te_time/dataset.num_te))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0", help='')
    parser.add_argument('--nn', type=int, default=0, help='')
    parser.add_argument('--ksize', type=int, default=3, help='')
    parser.add_argument('--lr', type=float, default=1e-4, help='')
    parser.add_argument('--batch', type=int, default=32, help='')
    parser.add_argument('--epochs', type=int, default=100, help='')

    FLAGS, unparsed = parser.parse_known_args()

    main()
