import os
import numpy as np
import tensorflow as tf
import source.utils as utils
import whiteboxlayer.layers as wbl
import whiteboxlayer.extensions.utility as wblu

class Agent(object):

    def __init__(self, **kwargs):

        print("\nInitializing Neural Network...")

        self.dim_h = kwargs['dim_h']
        self.dim_w = kwargs['dim_w']
        self.dim_c = kwargs['dim_c']
        self.num_class = kwargs['num_class']
        self.ksize = kwargs['ksize']
        self.learning_rate = kwargs['learning_rate']
        self.path_ckpt = kwargs['path_ckpt']

        self.variables = {}

        self.__model = Neuralnet(\
            who_am_i="CNN", **kwargs, \
            filters=[1, 32, 64, 128])

        dummy = tf.zeros((1, self.dim_h, self.dim_w, self.dim_c), dtype=tf.float32)
        self.__model.forward(x=dummy, verbose=True)

        self.__init_propagation(path=self.path_ckpt)

    def __init_propagation(self, path):

        self.summary_writer = tf.summary.create_file_writer(self.path_ckpt)

        self.variables['trainable'] = []
        ftxt = open("list_parameters.txt", "w")
        for key in list(self.__model.layer.parameters.keys()):
            trainable = self.__model.layer.parameters[key].trainable
            text = "T: " + str(key) + str(self.__model.layer.parameters[key].shape)
            if(trainable):
                self.variables['trainable'].append(self.__model.layer.parameters[key])
            ftxt.write("%s\n" %(text))
        ftxt.close()

        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.save_params()

        conc_func = self.__model.__call__.get_concrete_function(\
            tf.TensorSpec(shape=(1, self.dim_h, self.dim_w, self.dim_c), dtype=tf.float32))

    def __loss(self, y, y_hat):

        entropy_b = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
        entropy = tf.math.reduce_mean(entropy_b)

        return {'entropy_b': entropy_b, 'entropy': entropy}

    @tf.autograph.experimental.do_not_convert
    def step(self, minibatch, iteration=0, training=False):

        x, y = minibatch['x'], minibatch['y']

        with tf.GradientTape() as tape:
            attn, logit, y_hat = self.__model.forward(x=x, verbose=False)
            losses = self.__loss(y=y, y_hat=logit)

        if(training):
            gradients = tape.gradient(losses['entropy'], self.variables['trainable'])
            self.optimizer.apply_gradients(zip(gradients, self.variables['trainable']))

            with self.summary_writer.as_default():
                tf.summary.scalar('%s/entropy' %(self.__model.who_am_i), losses['entropy'], step=iteration)

        return {'attn':attn, 'y_hat':y_hat, 'losses':losses}

    def save_params(self, model='base', tflite=False):

        if(tflite):
            # https://github.com/tensorflow/tensorflow/issues/42818
            conc_func = self.__model.__call__.get_concrete_function(\
                tf.TensorSpec(shape=(1, self.dim_h, self.dim_w, self.dim_c), dtype=tf.float32))
            converter = tf.lite.TFLiteConverter.from_concrete_functions([conc_func])

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.experimental_new_converter = True
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

            tflite_model = converter.convert()

            with open('model.tflite', 'wb') as f:
                f.write(tflite_model)
        else:
            vars_to_save = self.__model.layer.parameters.copy()
            vars_to_save["optimizer"] = self.optimizer

            ckpt = tf.train.Checkpoint(**vars_to_save)
            ckptman = tf.train.CheckpointManager(ckpt, directory=os.path.join(self.path_ckpt, model), max_to_keep=1)
            ckptman.save()

    def load_params(self, model):

        vars_to_load = self.__model.layer.parameters.copy()
        vars_to_load["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_load)
        latest_ckpt = tf.train.latest_checkpoint(os.path.join(self.path_ckpt, model))
        status = ckpt.restore(latest_ckpt)
        status.expect_partial()

class Neuralnet(tf.Module):

    def __init__(self, **kwargs):
        super(Neuralnet, self).__init__()

        self.who_am_i = kwargs['who_am_i']
        self.dim_h = kwargs['dim_h']
        self.dim_w = kwargs['dim_w']
        self.dim_c = kwargs['dim_c']
        self.ksize = kwargs['ksize']
        self.num_class = kwargs['num_class']
        self.filters = kwargs['filters']

        self.layer = wbl.Layers()

        self.forward = tf.function(self.__call__)

    @tf.function
    def __call__(self, x, verbose=False):

        attn, logit = self.__nn(x=x, name=self.who_am_i, verbose=verbose)
        y_hat = tf.nn.softmax(logit, name="y_hat")

        return attn, logit, y_hat

    def __nn(self, x, name='neuralnet', verbose=True):

        att = None
        for idx, _ in enumerate(self.filters[:-1]):
            if(idx == 0): continue
            x = self.layer.conv2d(x=x, stride=1, \
                filter_size=[self.ksize, self.ksize, self.filters[idx-1], self.filters[idx]], \
                activation='relu', name='%s-%dconv1' %(name, idx), verbose=verbose)
            x = self.layer.conv2d(x=x, stride=1, \
                filter_size=[self.ksize, self.ksize, self.filters[idx], self.filters[idx]], \
                activation='relu', name='%s-%dconv2' %(name, idx), verbose=verbose)
            x = self.layer.maxpool(x=x, ksize=2, strides=2, \
                name='%s-%dmp' %(name, idx), verbose=verbose)
            if(idx == 2):
                attn = wblu.attention(self.layer.conv2d(x=x, stride=1, \
                    filter_size=[1, 1, self.filters[idx], self.dim_c], \
                    activation=None, name='%s-attn' %(name), verbose=verbose))
                x = x * attn

        x = self.layer.conv2d(x=x, stride=1, \
            filter_size=[self.ksize, self.ksize, self.filters[-2], 512], \
            activation='relu', name='%s-clf0' %(name), verbose=verbose)
        x = tf.math.reduce_mean(x, axis=(1, 2))
        x = self.layer.fully_connected(x=x, c_out=self.filters[-1], \
                activation='relu', name="%s-clf1" %(name), verbose=verbose)
        x = self.layer.fully_connected(x=x, c_out=self.num_class, \
                activation=None, name="%s-clf2" %(name), verbose=verbose)

        return attn, x
