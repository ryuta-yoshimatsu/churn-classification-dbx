from churn_classification_dbx.common import Task, MLflowTrackingConfig, FeatureStoreTableConfig, LabelsTableConfig
from churn_classification_dbx.model_train import ModelTrain, ModelTrainConfig
from churn_classification_dbx.utils.logger_utils import get_logger

_logger = get_logger()


class ModelTrainTask(Task):

    def _get_mlflow_tracking_cfg(self):
        try:
            experiment_id = self.env_vars['model_train_experiment_id']
        except KeyError:
            experiment_id = None
        try:
            experiment_path = self.env_vars['model_train_experiment_path']
        except KeyError:
            experiment_path = None

        return MLflowTrackingConfig(run_name=self.conf['mlflow_params']['run_name'],
                                    experiment_id=experiment_id,
                                    experiment_path=experiment_path,
                                    model_name=self.env_vars['model_name'])

    def _get_feature_store_table_cfg(self):
        return FeatureStoreTableConfig(database_name=self.env_vars['feature_store_database_name'],
                                       table_name=self.env_vars['feature_store_table_name'],
                                       primary_keys=self.env_vars['feature_store_table_primary_keys'])

    def _get_labels_table_cfg(self):
        return LabelsTableConfig(database_name=self.env_vars['labels_table_database_name'],
                                 table_name=self.env_vars['labels_table_name'],
                                 label_col=self.env_vars['labels_table_label_col'])

    def _get_pipeline_params(self):
        return self.conf['pipeline_params']

    def _get_model_params(self):
        return self.conf['model_params']

    def launch(self):
        _logger.info('Launching ModelTrainJob job')
        _logger.info(f'Running model-train pipeline in {self.env_vars["env"]} environment')
        cfg = ModelTrainConfig(mlflow_tracking_cfg=self._get_mlflow_tracking_cfg(),
                               feature_store_table_cfg=self._get_feature_store_table_cfg(),
                               labels_table_cfg=self._get_labels_table_cfg(),
                               pipeline_params=self._get_pipeline_params(),
                               model_params=self._get_model_params(),
                               conf=self.conf,
                               env_vars=self.env_vars)
        ModelTrain(cfg).run()
        _logger.info('ModelTrainJob job finished!')


# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = ModelTrainTask()
    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()
