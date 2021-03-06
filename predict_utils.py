import FLAGS
import tensorflow as tf
import os
from model_new.abuse_classifier import AbuseClassifier
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from data_utils_v2.data_helpers import genFeatures, loadVocabEmb,genPOSFeatures, loadData
import param
import pickle
from data_utils_v2 import eval_helpers
from sklearn.metrics import f1_score, accuracy_score
from  sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, RocCurveDisplay
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, HTML






def get_predictions(model_folder_path, data_folder_path, data_type="test", dump_folder_path="dump_2",
                    model_type="baseline"):
    with open(os.path.join(dump_folder_path, "norm_init_embed.pkl"), "rb") as handle:
        init_embed = pickle.load(handle)

    # configure model type
    if model_type == "baseline":
        attention_lambda = 0.0
        attention_loss_type = "none"
        model_type_path = "model_noatt_checkpoints"
    else:
        attention_lambda = 0.2
        attention_loss_type = "encoded"
        model_type_path = "model_att=encoded_checkpoints"

    checkpoint_dir = os.path.abspath(os.path.join(model_folder_path, model_type_path))
    model_path = os.path.join(checkpoint_dir, "best_model")
    print(f"Running {model_type} from {model_path}")

    x_test, length_test, attention_test, pos_test, pos_length_test, y_test = loadData(dump_folder_path,
                                                                                      data_folder_path, data_type,
                                                                                      verbose=False)

    len_data = len(x_test)
    print(f"Running model on {len_data} {data_type} samples")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = AbuseClassifier(
                max_sequence_length=param.max_sent_len,
                num_classes=2,
                pos_vocab_size=FLAGS.pos_vocab_size,
                init_embed=init_embed,
                hidden_size=FLAGS.hidden_size,
                attention_size=FLAGS.attention_size,
                keep_prob=FLAGS.dropout_keep_prob,
                attention_lambda=attention_lambda,
                attention_loss_type=attention_loss_type,
                l2_reg_lambda=0.1,
                use_pos_flag=FLAGS.use_pos_flag)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            saver = tf.train.Saver(tf.all_variables())
            # Initialize all variables
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, model_path)

            dev_scores = []
            dev_confidences = []
            alphas = []
            pos = 0
            gap = 50
            while pos < len(x_test):
                x_batch = x_test[pos:pos + gap]
                pos_batch = pos_test[pos:pos + gap]
                y_batch = y_test[pos:pos + gap]
                length_batch = length_test[pos:pos + gap]
                pos_length_batch = pos_length_test[pos:pos + gap]
                pos += gap
                # score sentences
                feed_dict = {
                    model.input_word: x_batch,
                    model.input_pos: pos_batch,
                    model.input_y: y_batch,
                    model.sequence_length: length_batch,
                    model.dropout_keep_prob: 1.0
                }
                step, scores, alpha = sess.run([global_step, model.prob, model.alphas], feed_dict)
                dev_confidences = dev_confidences + list([[s[0], s[1]] for s in scores])
                dev_scores = dev_scores + list([s[0] for s in scores])
                alphas = alphas + list(alpha)
    ground_truth = [g[0] for g in y_test]
    return dev_confidences, dev_scores, ground_truth, alphas


def get_metrics(ground_truth,confidences, best_threshold= 0.3, to_print = True, sample_predictions=True):
    """
    utility to get and print print metrics for a given set of ground truth and predictions
    :param ground_truth:
    :param confidences:
    :param best_threshold:
    :return:
    """
    print("Done")
    print("-" * 100)
    predictions = [s[0] for s in confidences]
    if sample_predictions:
        print("SAMPLES:")
        print("Predictions:")
        print(confidences[0:10])

        print("Ground Truth:")
        print(ground_truth[0:10])

        print("Classification:")

        y_pred = list(map(lambda p: 1 if p > best_threshold else 0, predictions))
        print(y_pred[0:10])

    if to_print:
        print("METRICS: \n")
    # roc auc
    roc_auc = eval_helpers.evalROC(ground_truth, predictions,to_print=to_print)
    # pr auc
    pr_auc = eval_helpers.evalPR(ground_truth, predictions,to_print=to_print)
    # f1 score
    f1score = f1_score(ground_truth, y_pred)
    if to_print:
        print("fscore: {}".format(f1score))
    accuracy = accuracy_score(ground_truth, y_pred)
    if to_print:
        print("accuracy: {}".format(accuracy))
    return roc_auc,pr_auc,f1score,accuracy
def plot_confusion_matrix(ground_truth, predictions, title,best_threshold= 0.3):
    print(f"----------{title}------------------")
    y_pred = [1 if y > best_threshold else 0 for y in predictions]
    cm = confusion_matrix(ground_truth, y_pred)
    ConfusionMatrixDisplay(cm).plot()

def plot_roc_auc(ground_truth,predictions,title):
    print(f"----------{title}------------------")
    plot_roc_auc(ground_truth, predictions)
def colorize(words, color_array):
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words
    cmap = matplotlib.cm.Reds
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    return colored_string
def plot_attention(tokenized_comment, attention_weights):
    len_comment = len(tokenized_comment)

    # if seperate:
    #     print("====FORWARD ATTENTION===")
    #     s = colorize(tokenized_comment, attention_weights[0:50])
    #     display(HTML(s))
    #     print("====BACKWORD ATTENTION===")
    #     s = colorize(tokenized_comment, attention_weights[50:])
    #     display(HTML(s))
    # else:

    s = colorize(tokenized_comment, attention_weights)
    display(HTML(s))

