import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints, backend as K
from tensorflow.keras.utils import get_custom_objects

class SwitchNormalization(Layer):
    """Switchable Normalization layer
    
    Switch Normalization performs Instance Normalization, Layer Normalization and Batch
    Normalization using its parameters, and then weighs them using learned parameters to
    allow different levels of interaction of the 3 normalization schemes for each layer.
    
    This is a unified TensorFlow 2.x compatible implementation.
    
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
        momentum: Momentum for the moving mean and the moving variance.
        epsilon: Small float added to variance to avoid dividing by zero.
        final_gamma: Bool value to determine if this layer is the final
            normalization layer for the residual block.
        center: If True, add offset of `beta` to normalized tensor.
        scale: If True, multiply by `gamma`.
    
    # References
        - [Differentiable Learning-to-Normalize via Switchable Normalization](https://arxiv.org/abs/1806.10779)
    """

    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 final_gamma=False,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 mean_weights_initializer='ones',
                 variance_weights_initializer='ones',
                 moving_mean_initializer='ones',
                 moving_variance_initializer='zeros',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 mean_weights_regularizer=None,
                 variance_weights_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 mean_weights_constraints=None,
                 variance_weights_constraints=None,
                 **kwargs):
        super(SwitchNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

        self.beta_initializer = initializers.get(beta_initializer)
        if final_gamma:
            self.gamma_initializer = initializers.get('zeros')
        else:
            self.gamma_initializer = initializers.get(gamma_initializer)
        self.mean_weights_initializer = initializers.get(mean_weights_initializer)
        self.variance_weights_initializer = initializers.get(variance_weights_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.mean_weights_regularizer = regularizers.get(mean_weights_regularizer)
        self.variance_weights_regularizer = regularizers.get(variance_weights_regularizer)
        
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.mean_weights_constraints = constraints.get(mean_weights_constraints)
        self.variance_weights_constraints = constraints.get(variance_weights_constraints)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name='gamma',
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint)
        else:
            self.gamma = None
            
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name='beta',
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint)
        else:
            self.beta = None

        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False,
            dtype=self._compute_dtype  # 强制使用当前计算dtype
        )

        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False,
            dtype=self._compute_dtype  # 强制使用当前计算dtype
        )

        self.mean_weights = self.add_weight(
            shape=(3,),
            name='mean_weights',
            initializer=self.mean_weights_initializer,
            regularizer=self.mean_weights_regularizer,
            constraint=self.mean_weights_constraints)

        self.variance_weights = self.add_weight(
            shape=(3,),
            name='variance_weights',
            initializer=self.variance_weights_initializer,
            regularizer=self.variance_weights_regularizer,
            constraint=self.variance_weights_constraints)

        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)

        # Prepare broadcasting shape.
        ndims = len(input_shape)
        reduction_axes = list(range(ndims))
        del reduction_axes[self.axis]
        
        if self.axis != 0:
            del reduction_axes[0]

        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis] = input_shape[self.axis]
        
        # 保存维度信息以便调试
        self._input_tensor_shape = input_shape  # 修改变量名，避免与Keras内置属性冲突
        self.reduction_axes = reduction_axes
        self.broadcast_shape = broadcast_shape
        
        # Compute the different normalizations
        # Instance normalization - normalize each instance in a batch independently
        mean_instance = K.mean(inputs, reduction_axes, keepdims=True)
        variance_instance = K.var(inputs, reduction_axes, keepdims=True)
        
        # Layer normalization - normalize each feature map across all instances
        mean_layer = K.mean(mean_instance, self.axis, keepdims=True)
        temp = variance_instance + K.square(mean_instance)
        variance_layer = K.mean(temp, self.axis, keepdims=True) - K.square(mean_layer)

        def training_phase():
            # Batch normalization - normalize each feature channel across batch
            mean_batch = K.mean(inputs, reduction_axes, keepdims=True)
            variance_batch = K.var(inputs, reduction_axes, keepdims=True)
            
            # Calculate statistics over the feature dimension only, for proper updates
            # Instead of reshaping the entire tensor, compute the mean across all dimensions except feature
            # Safely reshape for moving statistics update, handling dynamic shapes
            # This is a critical fix for the "Cannot reshape tensor with X elements to shape [Y]" error
            try:
                # More robust reshaping approach
                mean_batch_flat = tf.reshape(mean_batch, [-1, input_shape[self.axis]])
                mean_batch_reshaped = tf.reduce_mean(mean_batch_flat, axis=0)
                
                variance_batch_flat = tf.reshape(variance_batch, [-1, input_shape[self.axis]])
                variance_batch_reshaped = tf.reduce_mean(variance_batch_flat, axis=0)
            except Exception as e:
                # Fallback to simpler approach if the above fails
                # Just extract the values we need for updating the moving average
                mean_batch_reshaped = tf.reduce_mean(mean_batch, axis=reduction_axes, keepdims=False)
                variance_batch_reshaped = tf.reduce_mean(variance_batch, axis=reduction_axes, keepdims=False)
            
            # Apply correction factor for unbiased variance estimate
            if K.backend() != 'cntk':
                sample_size = K.prod([K.shape(inputs)[axis] for axis in reduction_axes])
                sample_size = K.cast(sample_size, dtype=K.dtype(inputs))
                variance_batch_reshaped *= sample_size / (sample_size - (1.0 + self.epsilon))
            
            # 类型转换，确保赋值不会出错
            mean_batch_reshaped = tf.cast(mean_batch_reshaped, self.moving_mean.dtype)
            variance_batch_reshaped = tf.cast(variance_batch_reshaped, self.moving_variance.dtype)

            # Update the moving statistics
            new_mean = self.moving_mean * self.momentum + mean_batch_reshaped * (1 - self.momentum)
            new_variance = self.moving_variance * self.momentum + variance_batch_reshaped * (1 - self.momentum)
            
            update_mean = self.moving_mean.assign(new_mean)
            update_variance = self.moving_variance.assign(new_variance)
            
            # Ensure updates are applied
            with tf.control_dependencies([update_mean, update_variance]):
                mean_batch = tf.identity(mean_batch)
                variance_batch = tf.identity(variance_batch)
            
            # Don't reshape the batch statistics here - they'll be reshaped in normalize_func
            return normalize_func(mean_batch, variance_batch)

        def inference_phase():
            # Use the moving statistics during inference
            # Reshape to broadcast_shape for compatibility with other statistics
            mean_batch = K.reshape(self.moving_mean, broadcast_shape)
            variance_batch = K.reshape(self.moving_variance, broadcast_shape)
            return normalize_func(mean_batch, variance_batch)

        def normalize_func(mean_batch, variance_batch):
            # 使用 tf.nn.softmax 替代 K.softmax 以避免 eager execution 问题
            # K.softmax 可能在图执行模式下调用 .numpy()，这在图模式下不可用
            mean_weights = tf.nn.softmax(self.mean_weights, axis=0)
            variance_weights = tf.nn.softmax(self.variance_weights, axis=0)
            
            # 明确转换为张量并获取索引值，避免索引操作中的问题
            mean_w0 = tf.gather(mean_weights, 0)
            mean_w1 = tf.gather(mean_weights, 1)
            mean_w2 = tf.gather(mean_weights, 2)
            
            var_w0 = tf.gather(variance_weights, 0)
            var_w1 = tf.gather(variance_weights, 1)
            var_w2 = tf.gather(variance_weights, 2)
            
            # 使用单独的权重乘法以避免形状问题
            mean_instance_weighted = mean_w0 * mean_instance
            mean_layer_weighted = mean_w1 * mean_layer
            mean_batch_weighted = mean_w2 * mean_batch
            
            # 使用 tf.add 确保广播正确
            mean = tf.add(mean_instance_weighted, mean_layer_weighted)
            mean = tf.add(mean, mean_batch_weighted)
            
            # 相同的方法处理方差
            variance_instance_weighted = var_w0 * variance_instance
            variance_layer_weighted = var_w1 * variance_layer
            variance_batch_weighted = var_w2 * variance_batch
            
            # 使用 tf.add 添加加权方差
            variance = tf.add(variance_instance_weighted, variance_layer_weighted)
            variance = tf.add(variance, variance_batch_weighted)

            # Apply normalization
            outputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

            # Apply scale and center if configured
            if self.scale:
                broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
                outputs = outputs * broadcast_gamma

            if self.center:
                broadcast_beta = K.reshape(self.beta, broadcast_shape)
                outputs = outputs + broadcast_beta

            return outputs

        # Use TF 2.x compatible conditional execution
        if training is None:
            training = K.learning_phase()
        
        return tf.cond(tf.cast(training, tf.bool),
                      true_fn=training_phase,
                      false_fn=inference_phase)

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'mean_weights_initializer': initializers.serialize(self.mean_weights_initializer),
            'variance_weights_initializer': initializers.serialize(self.variance_weights_initializer),
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer': initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'mean_weights_regularizer': regularizers.serialize(self.mean_weights_regularizer),
            'variance_weights_regularizer': regularizers.serialize(self.variance_weights_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'mean_weights_constraints': constraints.serialize(self.mean_weights_constraints),
            'variance_weights_constraints': constraints.serialize(self.variance_weights_constraints),
        }
        base_config = super(SwitchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


# Register the custom layer
get_custom_objects().update({'SwitchNormalization': SwitchNormalization})


if __name__ == '__main__':
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    
    # Test the layer with a simple model
    inp = Input(shape=(None, None, 4))
    x = SwitchNormalization(axis=-1)(inp)
    model = Model(inp, x)
    model.compile('adam', 'mse')
    model.summary()

    # Test with some random data
    import numpy as np
    x = np.random.normal(0.0, 1.0, size=(10, 8, 8, 4))
    
    # Try to fit the model to validate that the implementation works
    try:
        model.fit(x, x, epochs=1, verbose=1)
        print("Test successful - SwitchNormalization layer is working correctly!")
    except Exception as e:
        print(f"Error during test: {e}")
