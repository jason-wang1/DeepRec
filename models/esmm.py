import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l1, l2
from tensorflow.python.keras import losses, metrics
from layers.dnn import DNN
from layers.input_to_wide_emb import InputToWideEmb


class ESMM(Model):
    def __init__(self, config, **kwargs):
        self.config = config
        user_feat_list = config["model_config"]["feature_groups"]["user"]
        item_feat_list = config["model_config"]["feature_groups"]["item"]
        self.user_feat_config = [e for e in config["feature_config"]["features"] if e["input_names"] in user_feat_list]
        self.item_feat_config = [e for e in config["feature_config"]["features"] if e["input_names"] in item_feat_list]
        self.feat_size = len(config["feature_config"]["features"])
        self.emb_dim = config["model_config"]["embedding_dim"]
        self.dnn_shape = config["model_config"]["deep_hidden_units"]
        if "l2_reg" in config["model_config"]:
            self.reg = l2(config["model_config"]["l2_reg"])
        elif "l1_reg" in config["model_config"]:
            self.reg = l1(config["model_config"]["l1_reg"])
        else:
            self.reg = None
        self.batch_size = config['data_config']['batch_size']
        super(ESMM, self).__init__(**kwargs)
        self.loss_tracker = metrics.Mean(name="loss")
        self.ctr_loss_tracker = metrics.Mean(name="ctr_loss")
        self.ctcvr_loss_tracker = metrics.Mean(name="ctcvr_loss")
        self.ctr_var_tracker = metrics.Mean(name="ctr_var")
        self.ctcvr_var_tracker = metrics.Mean(name="ctcvr_var")
        self.ctr_auc = metrics.AUC(name="ctr_auc")
        self.ctcvr_auc = metrics.AUC(name="ctcvr_auc")

    def build(self, input_shape):
        self.user_input_to_wide_emb = InputToWideEmb(False, self.emb_dim, self.user_feat_config, self.reg, name="user_input")
        self.item_input_to_wide_emb = InputToWideEmb(False, self.emb_dim, self.item_feat_config, self.reg, name="item_input")
        self.ctr_tower = DNN(dnn_shape=self.dnn_shape, reg=self.reg, name="ctr_tower")
        self.cvr_tower = DNN(dnn_shape=self.dnn_shape, reg=self.reg, name="cvr_tower")

    def call(self, inputs, training=None, mask=None):
        user_emb_tensor = self.user_input_to_wide_emb(inputs)
        item_emb_tensor = self.item_input_to_wide_emb(inputs)
        user = tf.reduce_sum(user_emb_tensor, axis=1)  # (batch_size, emb_size)
        item = tf.reduce_sum(item_emb_tensor, axis=1)  # (batch_size, emb_size)
        dnn_input = tf.concat([user, item], axis=1)  # (batch_size, 2 * emb_size)
        ctr_pred = self.ctr_tower(dnn_input)  # (batch_size, 1)
        ctr_pred = tf.squeeze(tf.sigmoid(ctr_pred), axis=1)
        cvr_pred = self.cvr_tower(dnn_input)  # (batch_size, 1)
        cvr_pred = tf.squeeze(tf.sigmoid(cvr_pred), axis=1)
        ctcvr_pred = ctr_pred * cvr_pred
        return ctr_pred, ctcvr_pred

    def train_step(self, data):
        x, y = data
        ctr_label = y["click"]
        ctcvr_label = y["conversion"]
        ctr_label_float = tf.cast(ctr_label, dtype=tf.float32)
        ctcvr_label_float = tf.cast(ctcvr_label, dtype=tf.float32)
        ctr_s2 = 1 / (self.batch_size - 1) * tf.reduce_sum(tf.square(ctr_label_float - tf.reduce_mean(ctr_label_float))) + tf.constant(1.0*10**(-5))
        ctcvr_s2 = 1 / (self.batch_size - 1) * tf.reduce_sum(tf.square(ctcvr_label_float - tf.reduce_mean(ctcvr_label_float))) + tf.constant(1.0*10**(-5))

        with tf.GradientTape() as tape:
            ctr_pred, ctcvr_pred = self(x, training=True)  # Forward pass
            ctr_loss = losses.binary_crossentropy(ctr_label, ctr_pred)  # shape=()
            ctcvr_loss = losses.binary_crossentropy(ctcvr_label, ctcvr_pred)
            loss = (1.0/(2 * ctr_s2)) * ctr_loss + (1.0/(2 * ctcvr_s2)) * ctcvr_loss + tf.math.log(tf.sqrt(ctr_s2 * ctcvr_s2))

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.ctr_loss_tracker.update_state(ctr_loss)
        self.ctcvr_loss_tracker.update_state(ctcvr_loss)
        self.ctr_var_tracker.update_state(ctr_s2)
        self.ctcvr_var_tracker.update_state(ctcvr_s2)
        self.ctr_auc.update_state(ctr_label, ctr_pred)
        self.ctcvr_auc.update_state(ctcvr_label, ctcvr_pred)
        return {"loss": self.loss_tracker.result(), "ctr_loss": self.ctr_loss_tracker.result(),
                "ctcvr_loss": self.ctcvr_loss_tracker.result(),
                "ctr_s2": self.ctr_var_tracker.result(), "ctcvr_s2": self.ctcvr_var_tracker.result(),
                "ctr_auc": self.ctr_auc.result(), "ctcvr_auc": self.ctcvr_auc.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.ctr_loss_tracker, self.ctcvr_loss_tracker, self.ctr_var_tracker, self.ctcvr_var_tracker, self.ctr_auc, self.ctcvr_auc]
