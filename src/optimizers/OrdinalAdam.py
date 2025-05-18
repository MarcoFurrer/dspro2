import tensorflow as tf
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts, PolynomialDecay
from tensorflow.keras.optimizers import Adam
import math

# Advanced learning rate scheduler with warmup, plateau and cyclical phases
class AdvancedLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, 
                 initial_learning_rate=0.001,
                 min_learning_rate=0.00001,
                 warmup_steps=500,
                 decay_steps=5000,
                 plateau_steps=1000,
                 cycle_steps=2000):
        super(AdvancedLearningRateSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.plateau_steps = plateau_steps
        self.cycle_steps = cycle_steps
        
        # Cosine decay with restarts for the main phase
        self.cosine_decay = CosineDecayRestarts(
            initial_learning_rate=initial_learning_rate,
            first_decay_steps=cycle_steps,
            t_mul=1.5,       # Gradually increase cycle length
            m_mul=0.85,      # Gradually reduce max learning rate in each cycle
            alpha=min_learning_rate / initial_learning_rate
        )
        
        # Polynomial decay for plateau phase
        self.poly_decay = PolynomialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            end_learning_rate=min_learning_rate,
            power=0.5
        )
    
    def __call__(self, step):
        # Cast to float32
        step_float = tf.cast(step, tf.float32)
        
        # Warmup phase - linear warmup
        warmup_lr = self.initial_learning_rate * (step_float / self.warmup_steps)
        
        # Plateau phase - maintains a steady learning rate
        plateau_lr = self.initial_learning_rate
        
        # Main decay phase - cosine decay with restarts
        decay_step = step_float - self.warmup_steps - self.plateau_steps
        decay_lr = self.cosine_decay(decay_step)
        
        # Apply warmup for initial steps
        lr = tf.cond(
            step_float < self.warmup_steps,
            lambda: warmup_lr,
            lambda: tf.cond(
                step_float < (self.warmup_steps + self.plateau_steps),
                lambda: plateau_lr,
                lambda: decay_lr
            )
        )
        
        return lr
    
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "plateau_steps": self.plateau_steps,
            "cycle_steps": self.cycle_steps
        }

# Create advanced learning rate schedule for feature interaction
lr_schedule = AdvancedLearningRateSchedule(
    initial_learning_rate=0.0015,  # Slightly higher initial rate
    min_learning_rate=1e-6,
    warmup_steps=500,              # Longer warmup period
    plateau_steps=1000,            # Extended plateau for stable training
    cycle_steps=2000               # Cyclical updates to escape local minima
)

# Weight decay scheduler - gradually increase weight decay
class WeightDecayScheduler:
    def __init__(self, initial_weight_decay=1e-6, final_weight_decay=1e-4, decay_steps=5000):
        self.initial_weight_decay = initial_weight_decay
        self.final_weight_decay = final_weight_decay
        self.decay_steps = decay_steps
        
    def __call__(self, step):
        step_float = min(float(step), self.decay_steps)
        progress = step_float / self.decay_steps
        return self.initial_weight_decay + progress * (self.final_weight_decay - self.initial_weight_decay)

# Create weight decay scheduler
weight_decay_scheduler = WeightDecayScheduler(
    initial_weight_decay=1e-7,
    final_weight_decay=1e-4,
    decay_steps=7000
)

# Create advanced optimizer with lookahead for better convergence
class LookaheadOptimizer(tf.keras.optimizers.Optimizer, name='OrdinalAdam'):
    def __init__(self, optimizer, sync_period=5, slow_step_size=0.5, name="Lookahead", **kwargs):
        super(LookaheadOptimizer, self).__init__(name, **kwargs)
        self.optimizer = optimizer
        self._sync_period = sync_period
        self._slow_step_size = slow_step_size
        self._step_count = None
        self._optimizer_variables = optimizer.variables()
        self.name = name
        
    def _create_slots(self, var_list):
        self.optimizer._create_slots(var_list)
        for var in var_list:
            self.add_slot(var, "slow")
    
    def _prepare(self, var_list):
        return self.optimizer._prepare(var_list)
    
    @property
    def iterations(self):
        return self.optimizer.iterations
        
    def _resource_apply_dense(self, grad, var, apply_state=None):
        # Apply the internal optimizer update
        result = self.optimizer._resource_apply_dense(grad, var, apply_state)
        
        # Get the slot variable
        slow_var = self.get_slot(var, "slow")
        
        # Check if it's time to sync with slow weights
        should_sync = tf.equal(self.iterations % self._sync_period, 0)
        
        def _update_slow_var():
            # Compute the sync with slow weights when it's time
            slow_var_update = slow_var.assign_add(
                self._slow_step_size * (var - slow_var))
            # Update the variable with the slow variable
            var_update = var.assign(slow_var)
            return tf.group(slow_var_update, var_update)
        
        sync_op = tf.cond(
            should_sync,
            _update_slow_var,
            lambda: tf.no_op()
        )
        
        # Group the optimizer update and synchronization
        return tf.group(result, sync_op)
    
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # Apply the internal optimizer update
        result = self.optimizer._resource_apply_sparse(grad, var, indices, apply_state)
        
        # Get the slot variable
        slow_var = self.get_slot(var, "slow")
        
        # Check if it's time to sync with slow weights
        should_sync = tf.equal(self.iterations % self._sync_period, 0)
        
        def _update_slow_var():
            # Compute the sync with slow weights when it's time
            slow_var_update = slow_var.assign_add(
                self._slow_step_size * (var - slow_var))
            # Update the variable with the slow variable
            var_update = var.assign(slow_var)
            return tf.group(slow_var_update, var_update)
        
        sync_op = tf.cond(
            should_sync,
            _update_slow_var,
            lambda: tf.no_op()
        )
        
        # Group the optimizer update and synchronization
        return tf.group(result, sync_op)
    
    def get_config(self):
        config = {
            "sync_period": self._sync_period,
            "slow_step_size": self._slow_step_size,
            "optimizer": tf.keras.optimizers.serialize(self.optimizer),
        }
        base_config = super(LookaheadOptimizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Create Adam optimizer with weight decay
base_optimizer = Adam(
    learning_rate=lr_schedule,
    beta_1=0.9,             # Momentum parameter
    beta_2=0.999,           # RMSprop parameter
    epsilon=1e-7,           # Small constant for numerical stability
    amsgrad=True            # Use AMSGrad for more stable updates
)

# Wrap with Lookahead for improved convergence
optimizer = LookaheadOptimizer(
    optimizer=base_optimizer,
    sync_period=6,          # Synchronize every 6 steps
    slow_step_size=0.5      # Move halfway to the fast weights
)

class OrdinalAdam(Adam):
    """
    Enhanced Adam optimizer specifically tuned for ordinal regression models.
    Implements gradient clipping, cosine decay learning rate schedule with restarts,
    and adjustable parameters for better convergence with feature interactions.
    """
    
    def __init__(self, 
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 amsgrad=False,
                 clipnorm=1.0,
                 use_cosine_decay=True,
                 warmup_steps=500,
                 name="OrdinalAdam",
                 **kwargs):
        
        self.clipnorm = clipnorm
        self.use_cosine_decay = use_cosine_decay
        self.warmup_steps = warmup_steps
        
        # Create cosine decay schedule if enabled
        if use_cosine_decay:
            lr_schedule = CosineDecayRestarts(
                initial_learning_rate=learning_rate,
                first_decay_steps=2000,
                t_mul=2.0,  # Increase restart interval each time
                m_mul=0.85,  # Reduce max learning rate each restart
                alpha=0.1,   # Minimum learning rate factor
                name="cosine_decay_restarts"
            )
            
            # Apply warmup if specified
            if warmup_steps > 0:
                lr_schedule = WarmupSchedule(
                    lr_schedule,
                    warmup_steps=warmup_steps,
                    initial_learning_rate=learning_rate/10
                )
                
            learning_rate = lr_schedule
            
        super(OrdinalAdam, self).__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            clipnorm=clipnorm,
            name=name,
            **kwargs
        )
    
    def get_config(self):
        config = super(OrdinalAdam, self).get_config()
        config.update({
            "clipnorm": self.clipnorm,
            "use_cosine_decay": self.use_cosine_decay,
            "warmup_steps": self.warmup_steps
        })
        return config

# Learning rate warm-up schedule
class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Implements learning rate warmup schedule.
    Gradually increases learning rate from initial_learning_rate to main_schedule
    over warmup_steps, then follows main_schedule.
    """
    
    def __init__(self, main_schedule, warmup_steps, initial_learning_rate=0.0, name=None):
        super(WarmupSchedule, self).__init__()
        self.main_schedule = main_schedule
        self.warmup_steps = warmup_steps
        self.initial_learning_rate = initial_learning_rate
        self.name = name
    
    def __call__(self, step):
        with tf.name_scope(self.name or "WarmupSchedule"):
            # Convert to float
            step = tf.cast(step, tf.float32)
            warmup_steps = tf.cast(self.warmup_steps, tf.float32)
            
            # Warmup phase: linear increase
            warmup_factor = tf.minimum(step / warmup_steps, 1.0)
            warmup_lr = self.initial_learning_rate + warmup_factor * (self.main_schedule(0) - self.initial_learning_rate)
            
            # Switch between warmup and main schedule
            final_lr = tf.cond(
                step < warmup_steps,
                lambda: warmup_lr,
                lambda: self.main_schedule(step - warmup_steps)
            )
            
            return final_lr
    
    def get_config(self):
        return {
            "main_schedule": self.main_schedule,
            "warmup_steps": self.warmup_steps,
            "initial_learning_rate": self.initial_learning_rate,
            "name": self.name
        }