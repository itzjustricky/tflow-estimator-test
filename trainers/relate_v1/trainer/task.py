import argparse
import os

import model

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils)
from tensorflow.contrib.training.python.training import hparam


# TODO: must manipulate this accordingly
FEATURES = [
    # Categorical base columns

    # For categorical columns with known values we can provide lists
    # of values ahead of time.
    tf.feature_column.categorical_column_with_vocabulary_list(
        'gender', [' Female', ' Male']),

    tf.feature_column.categorical_column_with_vocabulary_list(
        'race',
        [' Amer-Indian-Eskimo', ' Asian-Pac-Islander',
         ' Black', ' Other', ' White']
    ),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'education',
        [' Bachelors', ' HS-grad', ' 11th', ' Masters', ' 9th',
         ' Some-college', ' Assoc-acdm', ' Assoc-voc', ' 7th-8th',
         ' Doctorate', ' Prof-school', ' 5th-6th', ' 10th',
         ' 1st-4th', ' Preschool', ' 12th']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status',
        [' Married-civ-spouse', ' Divorced', ' Married-spouse-absent',
         ' Never-married', ' Separated', ' Married-AF-spouse', ' Widowed']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship',
        [' Husband', ' Not-in-family', ' Wife', ' Own-child', ' Unmarried',
         ' Other-relative']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass',
        [' Self-emp-not-inc', ' Private', ' State-gov',
         ' Federal-gov', ' Local-gov', ' ?', ' Self-emp-inc',
         ' Without-pay', ' Never-worked']
    ),

    # For columns with a large number of values, or unknown values
    # We can use a hash function to convert to categories.
    tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=100, dtype=tf.string),
    tf.feature_column.categorical_column_with_hash_bucket(
        'native_country', hash_bucket_size=100, dtype=tf.string),

    # Continuous base columns.
    tf.feature_column.numeric_column('age'),
    tf.feature_column.numeric_column('education_num'),
    tf.feature_column.numeric_column('capital_gain'),
    tf.feature_column.numeric_column('capital_loss'),
    tf.feature_column.numeric_column('hours_per_week'),
]



def data_retrieval_fn(files):
    # merge files into single dataframe

    # prep the dataframe

    pass


def train_test_split(data_df):
    pass


def generate_input_fns(hparams):

    data_df = data_retrieval_fn(hparams.data_files)
    # split the data
    # data_df =
    (X_train, y_train), (X_test, y_test) = train_test_split(data_df)

    train_input = tf.estimators.inputs.pandas_input_fn(
        X_train, y_train,
        num_epochs=hparams.num_epochs,
        batch_size=hparams.train_batch_size,
    )
    eval_input = tf.estimators.inputs.pandas_input_fn(
        X_test, y_test,
        num_epochs=hparams.num_epochs,
        batch_size=hparams.train_batch_size,
        shuffle=False   # evaluation data should not be shuffled
    )

    return train_input, eval_input


def generate_experiment_fn(**experiment_args):
    """Create an experiment function.

    See command line help text for description of args.
    Args:
      experiment_args: keyword arguments to be passed through to experiment
        See `tf.contrib.learn.Experiment` for full args.
    Returns:
      A function:
        (tf.contrib.learn.RunConfig, tf.contrib.training.HParams) -> Experiment

      This function is used by learn_runner to create an Experiment which
      executes model code provided in the form of an Estimator and
      input functions.
    """
    def _experiment_fn(run_config, hparams):

        train_input, eval_input = generate_input_fns(hparams)

        # num_epochs can control duration if train_steps isn't
        # passed to Experiment
        # train_input = lambda: model.generate_input_fn(
        #     hparams.train_files,
        #     num_epochs=hparams.num_epochs,
        #     batch_size=hparams.train_batch_size,
        # )
        # # Don't shuffle evaluation data
        # eval_input = lambda: model.generate_input_fn(
        #     hparams.eval_files,
        #     batch_size=hparams.eval_batch_size,
        #     shuffle=False
        # )

        # TODO get rid of hidden_units & embedding size
        # build_estimator should be passed features
        return tf.contrib.learn.Experiment(
            model.build_estimator(
                embedding_size=hparams.embedding_size,
                # Construct layers sizes with exponetial decay
                hidden_units=[
                    max(2, int(hparams.first_layer_size *
                               hparams.scale_factor**i))
                    for i in range(hparams.num_layers)
                ],
                config=run_config
            ),
            train_input_fn=train_input,
            eval_input_fn=eval_input,
            **experiment_args
        )
    return _experiment_fn


def get_args():
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--data-files',
        help='GCS or local paths to training data',
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--num-epochs',
        help="""\
        Maximum number of training data epochs on which to train.
        If both --max-steps and --num-epochs are specified,
        the training job will run for --max-steps or --num-epochs,
        whichever occurs first. If unspecified will run for --max-steps.\
        """,
        type=int,
    )
    parser.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        type=int,
        default=40
    )
    parser.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=40
    )
    # parser.add_argument(
    #     '--eval-files',
    #     help='GCS or local paths to evaluation data',
    #     nargs='+',
    #     required=True
    # )
    # Training arguments
    parser.add_argument(
        '--embedding-size',
        help='Number of embedding dimensions for categorical columns',
        default=8,
        type=int
    )
    parser.add_argument(
        '--first-layer-size',
        help='Number of nodes in the first layer of the DNN',
        default=100,
        type=int
    )
    parser.add_argument(
        '--num-layers',
        help='Number of layers in the DNN',
        default=4,
        type=int
    )
    parser.add_argument(
        '--scale-factor',
        help='How quickly should the size of the layers in the DNN decay',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )

    # Argument to turn on all logging
    parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='INFO',
    )
    # Experiment arguments
    parser.add_argument(
        '--eval-delay-secs',
        help='How long to wait before running first evaluation',
        default=10,
        type=int
    )
    parser.add_argument(
        '--min-eval-frequency',
        help='Minimum number of training steps between evaluations',
        default=None,  # Use TensorFlow's default (currently, 1000 on GCS)
        type=int
    )
    parser.add_argument(
        '--train-steps',
        help="""\
        Steps to run the training job for. If --num-epochs is not specified,
        this must be. Otherwise the training job will run indefinitely.\
        """,
        type=int
    )
    parser.add_argument(
        '--eval-steps',
        help='Number of steps to run evalution for at each checkpoint',
        default=100,
        type=int
    )
    parser.add_argument(
        '--export-format',
        help='The input format of the exported SavedModel binary',
        choices=['JSON', 'CSV', 'EXAMPLE'],
        default='JSON'
    )

    return parser.parse_args()


def main():
    args = get_args()
    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.logging.__dict__[args.verbosity] / 10)

    # [START learn-runner]
    # Run the training job
    # learn_runner pulls configuration information from environment
    # variables using tf.learn.RunConfig and uses this configuration
    # to conditionally execute Experiment, or param server code
    learn_runner.run(
        generate_experiment_fn(
            min_eval_frequency=args.min_eval_frequency,
            eval_delay_secs=args.eval_delay_secs,
            train_steps=args.train_steps,
            eval_steps=args.eval_steps,
            export_strategies=[saved_model_export_utils.make_export_strategy(
                model.SERVING_FUNCTIONS[args.export_format],
                exports_to_keep=1,
                default_output_alternative_key=None,
            )]
        ),
        run_config=run_config.RunConfig(model_dir=args.job_dir),
        hparams=hparam.HParams(**args.__dict__)
    )
    # [END learn-runner]


if __name__ == '__main__':
    main()
