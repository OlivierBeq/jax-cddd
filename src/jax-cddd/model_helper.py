"""Helper functions that build the translation model with a corroponding graph and session."""
from collections import namedtuple
import tensorflow as tf
import models
import input_pipeline


def build_models(hparams):
    """Helper function to build a translation model for one or many different modes.

    Args:
        hparams: Hyperparameters defined in file or flags.
        modes: The mode the model is supposed to run (e.g. Train, EVAL, ENCODE, DECODE).
        Can be a list if multiple models should be build.
    Returns:
        One model or a list of multiple models.
    """
    model = getattr(models, "NoisyGRUSeq2SeqWithFeatures")
    model_list = []
    for mode in ["ENCODE", "DECODE"]:
        model_list.append(create_model(mode, model, hparams))
    return tuple(model_list)

Model = namedtuple("Model", ("graph", "model", "sess"))

def create_model(mode, model_creator, hparams):
    """Helper function to build a translation model for a certain mode.

    Args:cpu_threads
        mode: The mode the model is supposed to run(e.g. Train, EVAL, ENCODE, DECODE).
        model_creator: Type of model class (e.g. NoisyGRUSeq2SeqWithFeatures).
        hparams: Hyperparameters defined in file or flags.
    Returns:
        One model as named tuple with a graph, model and session object.
    """

    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        iterator = None
        model = model_creator(mode=mode,
                              iterator=iterator,
                              hparams=hparams
                             )
        model.build_graph()
        sess = tf.Session(graph=graph)
    return Model(graph=graph, model=model, sess=sess)
