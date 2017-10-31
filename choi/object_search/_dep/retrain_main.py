parser = argparse.ArgumentParser()
parser.add_argument(
    '--image_dir',
    type=str,
    default='',
    help='Path to folders of labeled images.'
)
parser.add_argument(
    '--output_graph',
    type=str,
    default='/tmp/output_graph.pb',
    help='Where to save the trained graph.'
)
parser.add_argument(
    '--output_labels',
    type=str,
    default='/tmp/output_labels.txt',
    help='Where to save the trained graph\'s labels.'
)
parser.add_argument(
    '--summaries_dir',
    type=str,
    default='/tmp/retrain_logs',
    help='Where to save summary logs for TensorBoard.'
)
parser.add_argument(
    '--how_many_training_steps',
    type=int,
    default=4000,
    help='How many training steps to run before ending.'
)
parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.01,
    help='How large a learning rate to use when training.'
)
parser.add_argument(
    '--testing_percentage',
    type=int,
    default=10,
    help='What percentage of images to use as a test set.'
)
parser.add_argument(
    '--validation_percentage',
    type=int,
    default=10,
    help='What percentage of images to use as a validation set.'
)
parser.add_argument(
    '--eval_step_interval',
    type=int,
    default=10,
    help='How often to evaluate the training results.'
)
parser.add_argument(
    '--train_batch_size',
    type=int,
    default=100,
    help='How many images to train on at a time.'
)
parser.add_argument(
    '--test_batch_size',
    type=int,
    default=-1,
    help="""\
  How many images to test on. This test set is only used once, to evaluate
  the final accuracy of the model after training completes.
  A value of -1 causes the entire test set to be used, which leads to more
  stable results across runs.\
  """
)
parser.add_argument(
    '--validation_batch_size',
    type=int,
    default=100,
    help="""\
  How many images to use in an evaluation batch. This validation set is
  used much more often than the test set, and is an early indicator of how
  accurate the model is during training.
  A value of -1 causes the entire validation set to be used, which leads to
  more stable results across training iterations, but may be slower on large
  training sets.\
  """
)
parser.add_argument(
    '--print_misclassified_test_images',
    default=False,
    help="""\
  Whether to print out a list of all misclassified test images.\
  """,
    action='store_true'
)
parser.add_argument(
    '--model_dir',
    type=str,
    default='/tmp/imagenet',
    help="""\
  Path to classify_image_graph_def.pb,
  imagenet_synset_to_human_label_map.txt, and
  imagenet_2012_challenge_label_map_proto.pbtxt.\
  """
)
parser.add_argument(
    '--bottleneck_dir',
    type=str,
    default='/tmp/bottleneck',
    help='Path to cache bottleneck layer values as files.'
)
parser.add_argument(
    '--final_tensor_name',
    type=str,
    default='final_result',
    help="""\
  The name of the output classification layer in the retrained graph.\
  """
)
parser.add_argument(
    '--flip_left_right',
    default=False,
    help="""\
  Whether to randomly flip half of the training images horizontally.\
  """,
    action='store_true'
)
parser.add_argument(
    '--random_crop',
    type=int,
    default=0,
    help="""\
  A percentage determining how much of a margin to randomly crop off the
  training images.\
  """
)
parser.add_argument(
    '--random_scale',
    type=int,
    default=0,
    help="""\
  A percentage determining how much to randomly scale up the size of the
  training images by.\
  """
)
parser.add_argument(
    '--random_brightness',
    type=int,
    default=0,
    help="""\
  A percentage determining how much to randomly multiply the training image
  input pixels up or down by.\
  """
)
FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

