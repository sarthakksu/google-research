import warnings
import tensorflow as tf
import numpy as np
from bam.tf_crf.layer import CRF
from bam.tf_crf.crf import crf_decode


def _forward_alg(feats, lens_, transitions, units, START_TAG=0, STOP_TAG=1):
    init_alphas_ = []
    for x in range(units):
        if x == START_TAG:
            init_alphas_.append(0.0)
        else:
            init_alphas_.append(-10000.0)
    init_alphas_ = np.array(init_alphas_)
    forward_var = np.zeros(
        shape=(
            feats.shape[0],
            feats.shape[1] + 1,
            feats.shape[2]
        )
    )

    forward_var[:, 0, :] = init_alphas_[None, :].repeat(feats.shape[0], axis=0)
    forward_var = tf.Variable(forward_var, dtype=transitions.dtype, trainable=False)
    transitions = tf.repeat(tf.reshape(transitions, (1, transitions.shape[0], transitions.shape[1])), feats.shape[0], 0)

    for i in range(feats.shape[1]):
        emit_score = feats[:, i, :]

        forward_cont = tf.transpose(tf.repeat(forward_var[:, i, :][:, :, None], [transitions.shape[2]], axis=-1),
                                    (0, 2, 1))
        emit_cont = tf.transpose(tf.repeat(emit_score[:, None, :], transitions.shape[2], axis=1), (0, 2, 1))
        tag_var = (emit_cont
                   + transitions
                   + forward_cont
                   )
        max_tag_var = tf.reduce_max(tag_var, axis=2)

        tag_var = tag_var - tf.repeat(max_tag_var[:, :, None], transitions.shape[2], axis=-1)
        agg_ = tf.math.log(tf.math.reduce_sum(tf.math.exp(tag_var), axis=2))

        forward_var[:, i + 1, :].assign(max_tag_var + agg_)

    return forward_var[:, 1:]


def _backward_alg(feats, lens_, transitions, units, T=1, START_TAG=0, STOP_TAG=1):
    bw_transitions = tf.transpose(transitions)
    reversed_feats = tf.zeros_like(feats)
    reversed_feats = tf.reverse_sequence(feats, lens_, seq_axis=1, batch_axis=0)
    init_alphas_ = []
    for x in range(units):
        if x == STOP_TAG:
            init_alphas_.append(0.0)
        else:
            init_alphas_.append(-1e12)
    init_alphas_ = np.array(init_alphas_)

    forward_var = np.zeros(
        shape=(
            reversed_feats.shape[0],
            reversed_feats.shape[1] + 1,
            reversed_feats.shape[2]
        )
    )

    forward_var[:, 0, :] = init_alphas_[None, :].repeat(reversed_feats.shape[0], axis=0)
    forward_var_ = tf.Variable(forward_var, dtype=transitions.dtype, trainable=False)
    transitions = tf.repeat(tf.reshape(bw_transitions, (1, bw_transitions.shape[0], bw_transitions.shape[1])),
                            reversed_feats.shape[0], 0)

    if T != 1:
        transitions = transitions / T
        reversed_feats = reversed_feats / T

    for i in range(reversed_feats.shape[1]):
        if i == 0:
            emit_score = tf.zeros_like(reversed_feats[:, 0, :])

        else:
            emit_score = reversed_feats[:, i - 1, :]
        forward_cont_ = tf.transpose(tf.repeat(forward_var_[:, i, :][:, :, None], [transitions.shape[2]], axis=-1),
                                     (0, 2, 1))
        emit_cont = tf.repeat(emit_score[:, None, :], transitions.shape[2], axis=1)

        tag_var = (emit_cont
                   + transitions
                   + forward_cont_
                   )
        max_tag_var = tf.reduce_max(tag_var, axis=-1)

        tag_var = tag_var - tf.repeat(max_tag_var[:, :, None], transitions.shape[2], axis=-1)

        agg_ = tf.math.log(tf.math.reduce_sum(tf.math.exp(tag_var), axis=2))

        forward_var_[:, i + 1, :].assign(max_tag_var + agg_)
    backward_var = tf.identity(forward_var_[:, 1:])
    new_backward_var = tf.reverse_sequence(backward_var, lens_, seq_axis=1, batch_axis=0)
    return new_backward_var

def distillation_loss(features, teacher_features, mask, T = 1, teacher_is_score=True,sentence_level_loss=True):
		# TODO: time with mask, and whether this should do softmax
  if teacher_is_score:
    #teacher_prob=F.softmax(teacher_features/T, dim=-1)
    teacher_prob = tf.nn.softmax(teacher_features/T, axis=-1,)
  else:
    teacher_prob=teacher_features
  #KD_loss = torch.nn.functional.kl_div(F.log_softmax(features/T, dim=-1), teacher_prob,reduction='none') * mask.unsqueeze(-1) * T * T
  KD_loss_fn = tf.keras.losses.KLDivergence(reduction=tf.losses.Reduction.NONE)
  #print(teacher_prob,features,T)
  KD_loss = KD_loss_fn(tf.nn.log_softmax(features/T,axis=-1),teacher_prob) * tf.cast(mask,tf.float32) * T * T
  #print(tf.reduce_sum(KD_loss))
  #print(tf.constant(KD_loss.shape[0].value,dtype=tf.float32))
  KD_loss = tf.reduce_sum(KD_loss)/tf.constant(KD_loss.shape[0].value,dtype=tf.float32)
  return KD_loss

class CustomCRF(CRF):
    def __init__(
            self,
            START_TAG: int = 1,
            STOP_TAG: int = 2,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.START_TAG = START_TAG
        self.STOP_TAG = STOP_TAG

    def get_viterbi_decoding(self, potentials, sequence_length):
        # decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`
        decoded_tags, best_score = super().get_viterbi_decoding(potentials, sequence_length)
        forward_score = _forward_alg(potentials, sequence_length, self.chain_kernel, units=self.units,
                                     START_TAG=self.START_TAG, STOP_TAG=self.STOP_TAG)
        backward_score = _backward_alg(potentials, sequence_length, self.chain_kernel, units=self.units, T=1,
                                       START_TAG=self.START_TAG, STOP_TAG=self.STOP_TAG)
        #print("reached custom crf get_viterbi_decoding")
        return decoded_tags, best_score, forward_score, backward_score
