import os, pdb
import tensorflow as tf
print(os.environ['TF_CPP_MIN_LOG_LEVEL'])
import configparser
from utils.pointnet_transform_nets import input_transform_net, feature_transform_net
import utils.pointnet_tf_util as pointnet_tf_util
import cPickle as pickle
import importlib
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))


class FrustumPointNet():
    def __init__(self, config_path):

        # load parameters
        parser = configparser.SafeConfigParser()
        parser.read(config_path)
        num_point = parser.getint('general', 'num_point')
        MODEL_PATH = parser.get('general', 'model_path')
        model = parser.get('general', 'model')
        MODEL = importlib.import_module(model)

        # placeholders and loss
        with tf.device('/gpu:'+str(0)):
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl = \
                MODEL.placeholder_inputs(1, num_point) #batch size 1
            is_training_pl = tf.placeholder(tf.bool, shape=())
            end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl,
                is_training_pl)
            loss = MODEL.get_loss(labels_pl, centers_pl,
                heading_class_label_pl, heading_residual_label_pl,
                size_class_label_pl, size_residual_label_pl, end_points)
            self.saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

        # Restore variables from disk.
        self.saver.restore(sess, MODEL_PATH)
        self.ops = {'pointclouds_pl': pointclouds_pl,
               'one_hot_vec_pl': one_hot_vec_pl,
               'labels_pl': labels_pl,
               'centers_pl': centers_pl,
               'heading_class_label_pl': heading_class_label_pl,
               'heading_residual_label_pl': heading_residual_label_pl,
               'size_class_label_pl': size_class_label_pl,
               'size_residual_label_pl': size_residual_label_pl,
               'is_training_pl': is_training_pl,
               'logits': end_points['mask_logits'],
               'center': end_points['center'],
               'end_points': end_points,
               'loss': loss}

    ''' what is the format of PC??? '''
    ''' one_hot_vec is one hot encoding of the class of object (car, pedestrian, cyclist) '''
    ''' This code is originally written to process batches. Since we are doing frame by 
        frame, I have replaced those all with 1. We can later reformat to just collapse
        the dimension designated for batch sizes.'''
    def __call__(self, pc, one_hot_vec):

        # some constants
        NUM_CLASSES = 2
        NUM_CHANNEL = 4
        NUM_HEADING_BIN = 12
        NUM_SIZE_CLUSTER = 8 # one cluster for each type

        logits = np.zeros((pc.shape[0], pc.shape[1], NUM_CLASSES))
        centers = np.zeros((pc.shape[0], 3))
        heading_logits = np.zeros((pc.shape[0], NUM_HEADING_BIN))
        heading_residuals = np.zeros((pc.shape[0], NUM_HEADING_BIN))
        size_logits = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER))
        size_residuals = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER, 3))
        scores = np.zeros((pc.shape[0],)) # 3D box score 

        ep = ops['end_points'] 

        feed_dict = {\
            ops['pointclouds_pl']: pc[0:1,...],
            ops['one_hot_vec_pl']: one_hot_vec[0:1,:],
            ops['is_training_pl']: False}

        # first dimension originally used for batch sizes
        logits[0:1,...], centers[0:1,...], \
        heading_scores[0:1,...], heading_residuals[0:1,...], \
        size_scores[0:1,...], size_residuals[0:1,...] = \
            sess.run([ops['logits'], ops['center'],
                ep['heading_scores'], ep['heading_residuals'],
                ep['size_scores'], ep['size_residuals']],
                feed_dict=feed_dict)

        # Compute scores
        batch_seg_prob = softmax(logs[0:1,...])[:,:,1] # BxN
        batch_seg_mask = np.argmax(logits[0:1,...], 2) # BxN
        mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1) # B,
        mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask,1) # B,
        heading_prob = np.max(softmax(heading_scores[0,1:...]),1) # B
        size_prob = np.max(softmax(size_scores[0:1,...]),1) # B,
        scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)
        scores[0:1] = batch_scores 
        # Finished computing scores

        heading_cls = np.argmax(heading_logits, 1) # B
        size_cls = np.argmax(size_logits, 1) # B
        heading_res = np.array([heading_residuals[i,heading_cls[i]] \
            for i in range(pc.shape[0])])
        size_res = np.vstack([size_residuals[i,size_cls[i],:] \
            for i in range(pc.shape[0])])

        # Commented lines below process this output.
        # TODO: Modify to fix format as needed.
        return np.argmax(logits, 2), centers, heading_cls, heading_res, \
            size_cls, size_res, scores   

        '''
        ps_list = []
        segp_list = []
        center_list = []
        heading_cls_list = []
        heading_res_list = []
        size_cls_list = []
        size_res_list = []
        rot_angle_list = []
        score_list = []
        onehot_list = []

        ps_list.append(batch_data[0,...])
        segp_list.append(batch_output[0,...])
        center_list.append(batch_center_pred[0,:])
        heading_cls_list.append(batch_hclass_pred[0])
        heading_res_list.append(batch_hres_pred[0])
        size_cls_list.append(batch_sclass_pred[0])
        size_res_list.append(batch_sres_pred[0,:])
        rot_angle_list.append(batch_rot_angle[0])
        score_list.append(batch_scores[0])
        score_list.append(batch_rgb_prob[0]) # 2D RGB detection score
        onehot_list.append(batch_one_hot_vec[0])
        '''

    def softmax(x):
        ''' Numpy function for softmax'''
        shape = x.shape
        probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
        probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
        return probs    