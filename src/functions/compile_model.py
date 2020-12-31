from tensorflow import keras
from lib.u2net import *
from functions.training.dataloader import format_input
from PIL import Image
from tensorflow.python.framework import graph_io
from tensorflow.compat.v1.keras import backend as K
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow_model_optimization as tfmot
import tensorflow as tf
import shutil, os, numpy as np, time

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()


class CompileModel:

    def __init__(self, weights_file = None, dataset = None):
        self.weights_file = weights_file
        self.dataset = dataset
        
        self.optimize_model()
        #self.convert_model_to_tensorflow()
        #self.convert_to_tensorflow_lite()
        self.convert_model_to_tensorrt()

        #self.test_model_tensorflow()
        #self.test_model_tensorflow_lite()

    def optimize_model(self):
        inputs = keras.Input(shape=(256, 256, 3))
        net = U2NET()
        out = net(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=out, name='u2netmodel')
        model.compile(optimizer=adam, loss=bce_loss, metrics=None)
        model.load_weights(self.weights_file)
        model.summary()

        # Freeze model
        outputs = [ out.op.name for out in model.outputs ]
        session = K.get_session()
        with session.graph.as_default():
            graphdef_inf = tf.compat.v1.graph_util.remove_training_nodes(session.graph.as_graph_def())
            graphdef_frozen = tf.compat.v1.graph_util.convert_variables_to_constants(session, graphdef_inf, outputs)
            graph_io.write_graph(graphdef_frozen, 'data/model/frozen/', 'rcnn_mask.pb', as_text=False)
            self.frozen_model = graphdef_frozen

        self.model = model

    def convert_model_to_tensorflow(self):
        dir = os.path.join(self.dataset, "..", "model", "unoptimized", "tensorflow")
        try: shutil.rmtree(dir)
        except: pass
        os.makedirs(dir, exist_ok = True)
        self.model.save(dir)

    def convert_to_tensorflow_lite(self):
        dir = os.path.join(self.dataset, "..", "model", "unoptimized", "tensorflow-lite")
        keras_model = os.path.join(self.dataset, "..", "model", "unoptimized", "tensorflow")
        try: shutil.rmtree(dir)
        except: pass
        os.makedirs(dir, exist_ok = True)
        converter = tf.lite.TFLiteConverter.from_saved_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        tflite_model = converter.convert()
        with open(os.path.join(dir, "u2net.tflite"), "wb") as f:
            f.write(tflite_model)

    def convert_model_to_tensorrt(self):
        dir = os.path.join(self.dataset, "..", "model", "unoptimized", "tensorrt")
        try: shutil.rmtree(dir)
        except: pass
        os.makedirs(dir, exist_ok = True)

        outputs = [ out.op.name for out in self.model.outputs ]
        converter = trt.TrtGraphConverter(
            input_graph_def=self.frozen_model,
            precision_mode='FP32',
        )
        frozen_graph = converter.convert()
        frozen_graph = converter.calibrate(
            fetch_names=['logits', 'classes'],
            num_runs=num_calib_inputs // batch_size,
            input_map_fn=input_map_fn
        )

        converted_graph_def = trt.TrtGraphConverter(trt_graph)
        graph_io.write_graph(converted_graph_def, 'data/model/tensorrt/', 'rcnn_mask.pb', as_text=False)

    def test_model_tensorflow(self):
        model = keras.models.load_model(os.path.join(self.dataset, "..", "model", "unoptimized", "tensorflow"), custom_objects = {
            "bce_loss": bce_loss
        })
        model.summary()

        for _dir, _subdir, files in os.walk(os.path.join(self.dataset, "images")):
            image = Image.open(os.path.join(self.dataset, "images", files[0]))
            image = format_input(image)
            for i in range(10):
                s = time.time()
                preds = model.predict(image)
                e = time.time()
                print('[ OK ] Inference time: {:,.2f} || FPS: {:,.2f}'.format(e - s, 1 / (e - s)))

    def test_model_tensorflow_lite(self):
        model = tf.lite.Interpreter(model_path=os.path.join(self.dataset, "..", "model", "unoptimized", "tensorflow-lite", "u2net.tflite"))
        model.allocate_tensors()

        input_details = model.get_input_details()
        output_details = model.get_output_details()

        for _dir, _subdir, files in os.walk(os.path.join(self.dataset, "images")):
            image = Image.open(os.path.join(self.dataset, "images", files[0]))
            image = format_input(image, _type = 'float32')
            for i in range(10):
                s = time.time()
                model.allocate_tensors()
                preds = model.set_tensor(input_details[0]['index'], image)
                model.invoke()
                output_data = model.get_tensor(output_details[0]['index'])
                e = time.time()
                print('[ OK ] Inference time: {:,.2f}s || FPS: {:,.2f}'.format(e - s, 1 / (e - s)))

