import tensorflow as tf
import os

_FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

    def __init__(self, model_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = tf.GraphDef()

        frozen_graph = os.path.join(model_path, _FROZEN_GRAPH_NAME)
        with tf.gfile.GFile(frozen_graph, 'rb') as f:
            graph_def = graph_def.FromString(f.read())

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
            image: raw input image.

        Returns:
            seg_map: Segmentation map of input image.
        """
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [image]})
        seg_map = batch_seg_map[0]
        return seg_map
