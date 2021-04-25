import warnings
import tensorflow.compat.v1 as tf
import numpy as np
import tensorflow_addons as tfa
from tensorflow_addons.utils.types import TensorLike
from tensorflow_addons.text.crf import crf_decode_forward
from tensorflow_addons.text.crf import crf_decode_backward
from tf_slice_assign import slice_assign


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
    transitions = tf.repeat(tf.reshape(transitions, (1, transitions.shape[0], transitions.shape[1])), feats.shape[0], 0)

    for i in range(feats.shape[1]):
        emit_score = feats[:, i, :].numpy()

        forward_cont = forward_var[:, i, :][:, :, None].repeat([transitions.shape[2]], axis=-1).transpose(0, 2, 1)
        emit_cont = emit_score[:, None, :].repeat(transitions.shape[2], axis=1).transpose(0, 2, 1)
        tag_var = (emit_cont
                   + transitions
                   + forward_cont
                   )
        max_tag_var = tf.reduce_max(tag_var, reduction_indices=[2])

        tag_var = tag_var - tf.repeat(max_tag_var[:, :, None], transitions.shape[2], axis=-1)
        agg_ = tf.math.log(tf.math.reduce_sum(tf.math.exp(tag_var), axis=2))

        cloned = np.copy(forward_var)
        cloned[:, i + 1, :] = max_tag_var + agg_

        forward_var = cloned
    return forward_var[:, 1:]


def _backward_alg(feats, lens_, transitions, units, T=1, START_TAG=0, STOP_TAG=1):
    bw_transitions = tf.transpose(transitions)
    reversed_feats = tf.zeros_like(feats)

    new_reverse_feats = tf.TensorArray(dtype=feats.dtype, size=feats.shape[0])
    for i, feat in enumerate(feats):
        # m * d -> k * d, reverse over tokens -> m * d
        rev = slice_assign(reversed_feats[i], tf.reverse(feat[:lens_[i]], [0]), slice(None, lens_[i]))
        new_reverse_feats.write(i, rev)
        # reversed_feats[i][:lens_[i]] = tf.reverse(feat[:lens_[0]],[0])
        # reverse_feats[i][:lens_[i]] = feat[:lens_[i]].filp(0)
    reversed_feats = new_reverse_feats.stack()

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

    transitions = tf.repeat(tf.reshape(bw_transitions, (1, bw_transitions.shape[0], bw_transitions.shape[1])),
                            reversed_feats.shape[0], 0)

    if T != 1:
        transitions = transitions / T
        reversed_feats = reversed_feats / T

    for i in range(reversed_feats.shape[1]):
        if i == 0:
            emit_score = np.zeros_like(reversed_feats[:, 0, :])
        else:
            emit_score = reversed_feats[:, i - 1, :].numpy()
        # pdb.set_trace()
        forward_cont = forward_var[:, i, :][:, :, None].repeat([transitions.shape[2]], axis=-1).transpose(0, 2, 1)
        emit_cont = emit_score[:, None, :].repeat(transitions.shape[2], axis=1)
        tag_var = (emit_cont
                   + transitions
                   + forward_cont
                   )

        max_tag_var = tf.reduce_max(tag_var, reduction_indices=[-1])

        tag_var = tag_var - tf.repeat(max_tag_var[:, :, None], transitions.shape[2], axis=-1)

        agg_ = tf.math.log(tf.math.reduce_sum(tf.math.exp(tag_var), axis=2))

        cloned = np.copy(forward_var)
        cloned[:, i + 1, :] = max_tag_var + agg_

        forward_var = cloned
    backward_var = np.copy(forward_var[:, 1:])
    new_backward_var = np.zeros_like(backward_var)
    # for i, var in enumerate(backward_var):

    # flip over tokens, [num_tokens * num_tags]
    #  new_backward_var[i,:lens_[i]] = var[:lens_[i]].flip([0])

    tf_array = tf.TensorArray(dtype=backward_var.dtype, size=feats.shape[0])
    for i, var in enumerate(backward_var):
        # m * d -> k * d, reverse over tokens -> m * d
        rev = slice_assign(new_backward_var[i], tf.reverse(var[:lens_[i]], [0]), slice(None, lens_[i]))
        tf_array.write(i, rev)
        # reversed_feats[i][:lens_[i]] = tf.reverse(feat[:lens_[0]],[0])
        # reverse_feats[i][:lens_[i]] = feat[:lens_[i]].filp(0)
    new_backward_var = tf_array.stack()
    return new_backward_var


def crf_decode(
        potentials: TensorLike, transition_params: TensorLike, sequence_length: TensorLike, units,
        START_TAG=0, STOP_TAG=1) -> tf.Tensor:
    """Decode the highest scoring sequence of tags.
    Args:
      potentials: A [batch_size, max_seq_len, num_tags] tensor of
                unary potentials.
      transition_params: A [num_tags, num_tags] matrix of
                binary potentials.
      sequence_length: A [batch_size] vector of true sequence lengths.
    Returns:
      decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
                  Contains the highest scoring tag indices.
      best_score: A [batch_size] vector, containing the score of `decode_tags`.
    """
    if tf.__version__[:3] == "2.4":
        warnings.warn(
            "CRF Decoding does not work with KerasTensors in TF2.4. The bug has since been fixed in tensorflow/tensorflow##45534"
        )
    sequence_length = tf.cast(sequence_length, dtype=tf.int32)

    # If max_seq_len is 1, we skip the algorithm and simply return the
    # argmax tag and the max activation.
    def _single_seq_fn():
        decode_tags = tf.cast(tf.argmax(potentials, axis=2), dtype=tf.int32)
        best_score = tf.reshape(tf.reduce_max(potentials, axis=2), shape=[-1])
        return decode_tags, best_score, None, None

    def _multi_seq_fn():
        # Computes forward decoding. Get last score and backpointers.
        initial_state = tf.slice(potentials, [0, 0, 0], [-1, 1, -1])
        initial_state = tf.squeeze(initial_state, axis=[1])
        inputs = tf.slice(potentials, [0, 1, 0], [-1, -1, -1])

        sequence_length_less_one = tf.maximum(
            tf.constant(0, dtype=tf.int32), sequence_length - 1
        )

        backpointers, last_score = crf_decode_forward(
            inputs, initial_state, transition_params, sequence_length_less_one
        )

        backpointers = tf.reverse_sequence(
            backpointers, sequence_length_less_one, seq_axis=1
        )

        initial_state = tf.cast(tf.argmax(last_score, axis=1), dtype=tf.int32)
        initial_state = tf.expand_dims(initial_state, axis=-1)

        decode_tags = crf_decode_backward(backpointers, initial_state)

        decode_tags = tf.squeeze(decode_tags, axis=[2])
        decode_tags = tf.concat([initial_state, decode_tags], axis=1)
        decode_tags = tf.reverse_sequence(decode_tags, sequence_length, seq_axis=1)

        best_score = tf.reduce_max(last_score, axis=1)
        backward_score = _backward_alg(potentials, sequence_length, transition_params, units=units, START_TAG=START_TAG,
                                       STOP_TAG=STOP_TAG)
        forward_score = _forward_alg(potentials, sequence_length, transition_params, units=units, START_TAG=START_TAG,
                                     STOP_TAG=STOP_TAG)
        return decode_tags, best_score, forward_score, backward_score

    if potentials.shape[1] is not None:
        # shape is statically know, so we just execute
        # the appropriate code path
        if potentials.shape[1] == 1:
            return _single_seq_fn()
        else:
            return _multi_seq_fn()
    else:
        return tf.cond(
            tf.equal(tf.shape(potentials)[1], 1), _single_seq_fn, _multi_seq_fn
        )


class CustomCRF(tfa.layers.CRF):
    def __init__(
            self,
            START_TAG: int = 1,
            STOP_TAG: int = 2,
            **kwargs,
    ):
        super(CustomCRF,self).__init__(user_kernel=False,**kwargs)
        self.START_TAG = START_TAG
        self.STOP_TAG = STOP_TAG

    def call(self, inputs, mask=None):
        # mask: Tensor(shape=(batch_size, sequence_length), dtype=bool) or None

        if mask is not None:
            if len(mask.shape) != 2:
                raise ValueError("Input mask to CRF must have dim 2 if not None")

        if mask is not None:
            # left padding of mask is not supported, due the underline CRF function
            # detect it and report it to user
            left_boundary_mask = self._compute_mask_left_boundary(mask)
            first_mask = left_boundary_mask[:, 0]
            if first_mask is not None and tf.executing_eagerly():
                no_left_padding = tf.math.reduce_all(first_mask)
                left_padding = not no_left_padding
                if left_padding:
                    raise NotImplementedError(
                        "Currently, CRF layer do not support left padding"
                    )

        potentials = tf.layers.dense(inputs,self.units)#self._dense_layer(inputs)

        # appending boundary probability info
        if self.use_boundary:
            potentials = self.add_boundary_energy(
                potentials, mask, self.left_boundary, self.right_boundary
            )

        sequence_length = self._get_sequence_length(inputs, mask)

        decoded_sequence, best_score, forward_score, backward_score = self.get_viterbi_decoding(potentials,
                                                                                                sequence_length)

        return [decoded_sequence, potentials, sequence_length, self.chain_kernel, best_score, forward_score,
                backward_score]

    def get_viterbi_decoding(self, potentials, sequence_length):
        # decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`
        decode_tags, best_score, forward_score, backward_score = crf_decode(
            potentials, self.chain_kernel, sequence_length, units=self.units, START_TAG=self.START_TAG,
            STOP_TAG=self.STOP_TAG
        )

        return decode_tags, best_score, forward_score, backward_score
def distillation_loss(features, teacher_features, mask, T = 1, teacher_is_score=True,sentence_level_loss=True):
		# TODO: time with mask, and whether this should do softmax
  if teacher_is_score:
    #teacher_prob=F.softmax(teacher_features/T, dim=-1)
    teacher_prob = tf.nn.softmax(teacher_features/T, axis=-1,)
  else:
    teacher_prob=teacher_features
  #KD_loss = torch.nn.functional.kl_div(F.log_softmax(features/T, dim=-1), teacher_prob,reduction='none') * mask.unsqueeze(-1) * T * T
  KD_loss_fn = tf.keras.losses.KLDivergence(reduction=tf.losses.Reduction.NONE)
  KD_loss = KD_loss_fn(tf.nn.log_softmax(features/T,axis=-1),teacher_prob) * mask * T * T
  KD_loss = tf.reduce_sum(KD_loss)/KD_loss.shape[0]
  return KD_loss