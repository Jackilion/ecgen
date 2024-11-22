import optax



def create_learning_rate_fn(epochs, steps_per_epoch, base_learning_rate, max_learning_rate, warmup_epochs):
  """
    Creates learning rate schedule.
  """

  warmup_fn = optax.linear_schedule(
      init_value=0., end_value=max_learning_rate,
      transition_steps=warmup_epochs * steps_per_epoch)
  
  max_schedule = optax.constant_schedule(max_learning_rate)
  lowering_schedule = optax.linear_schedule(
      init_value=max_learning_rate,
    end_value=base_learning_rate,
    transition_steps=(epochs - warmup_epochs * 3) * steps_per_epoch
  )
#   cosine_epochs = max(FLAGS.AE_epochs - FLAGS.AE_warmup_epochs, 1)
#   cosine_fn = optax.cosine_decay_schedule(
#       init_value=FLAGS.AE_base_learning_rate,
#       decay_steps=cosine_epochs * steps_per_epoch)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, max_schedule, lowering_schedule],
      boundaries=[warmup_epochs * steps_per_epoch, 2* warmup_epochs * steps_per_epoch])
  return schedule_fn

def create_annealing_learning_rate_fn(total_epochs, steps_per_epoch, main_learning_rate):
  
  main_schedule = optax.constant_schedule(main_learning_rate)
  lowering_schedule = optax.linear_schedule(init_value=main_learning_rate, end_value=0.0,
                                            transition_steps=10 * steps_per_epoch)
  
  schedule_fn = optax.join_schedules(schedules=[main_schedule, lowering_schedule],
                                     boundaries=[(total_epochs - 20) * steps_per_epoch]
                                     )
  return schedule_fn