import optax


from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_float("AE_base_learning_rate", 1e-5, "The base learning rate")
flags.DEFINE_float("AE_max_learning_rate", 1e-4, "The max learning rate")
flags.DEFINE_float("AE_warmup_epochs", 5, "Number of epochs to warm up the lr schedule")



def create_learning_rate_fn(steps_per_epoch):
  """Creates learning rate schedule."""
  warmup_fn = optax.linear_schedule(
      init_value=0., end_value=FLAGS.AE_max_learning_rate,
      transition_steps=FLAGS.AE_warmup_epochs * steps_per_epoch)
  
  max_schedule = optax.constant_schedule(FLAGS.AE_max_learning_rate)
  lowering_schedule = optax.linear_schedule(
      init_value=FLAGS.AE_max_learning_rate,
    end_value=FLAGS.AE_base_learning_rate,
    transition_steps=(FLAGS.AE_epochs - FLAGS.AE_warmup_epochs * 3) * steps_per_epoch
  )
#   cosine_epochs = max(FLAGS.AE_epochs - FLAGS.AE_warmup_epochs, 1)
#   cosine_fn = optax.cosine_decay_schedule(
#       init_value=FLAGS.AE_base_learning_rate,
#       decay_steps=cosine_epochs * steps_per_epoch)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, max_schedule, lowering_schedule],
      boundaries=[FLAGS.AE_warmup_epochs * steps_per_epoch, 2* FLAGS.AE_warmup_epochs * steps_per_epoch])
  return schedule_fn

def create_annealing_learning_rate_fn(total_epochs, steps_per_epoch):
  
  main_schedule = optax.constant_schedule(FLAGS.DDIM_learning_rate)
  lowering_schedule = optax.linear_schedule(init_value=FLAGS.DDIM_learning_rate, end_value=0.0,
                                            transition_steps=10 * steps_per_epoch)
  
  schedule_fn = optax.join_schedules(schedules=[main_schedule, lowering_schedule],
                                     boundaries=[(total_epochs - 20) * steps_per_epoch]
                                     )
  return schedule_fn