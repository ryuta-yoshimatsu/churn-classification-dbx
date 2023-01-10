from dataclasses import dataclass

import pyspark.sql.dataframe

from churn_classification_dbx.common import FeatureStoreTableConfig, LabelsTableConfig
from churn_classification_dbx.featurize import Featurizer, FeaturizerConfig
from churn_classification_dbx.utils import feature_store_utils
from churn_classification_dbx.utils.get_spark import spark
from churn_classification_dbx.utils.logger_utils import get_logger

_logger = get_logger()


@dataclass
class FeatureTableRefresherConfig:
    """
    Attributes:
        input_table (str):
            Name of the table to use as input for refreshing features
        featurizer_cfg (FeaturizerConfig):
            Featurization config to specify label_col, ohe, cat_cols and drop_missing params
        feature_store_table_cfg (FeatureStoreTableConfig):
            Feature Store table config to specify database_name, table_name, primary_keys and description
        labels_table_cfg (LabelsTableConfig):
            Labels table config to specify database_name, table_name, label_col and dbfs_path
    """
    input_table: str
    featurizer_cfg: FeaturizerConfig
    feature_store_table_cfg: FeatureStoreTableConfig
    labels_table_cfg: LabelsTableConfig


class FeatureTableRefresher:
    """
    Class to execute a pipeline to refresh a Feature Store table, and separate labels table
    """
    def __init__(self, cfg: FeatureTableRefresherConfig):
        self.cfg = cfg

    @staticmethod
    def setup(database_name: str) -> None:
        """
        Set up database to use. Create the database {database_name} if it doesn't exist

        Parameters
        ----------
        database_name : str
            Database to create if it doesn't exist. Otherwise use database of the name provided
        """
        _logger.info(f'Creating database {database_name} if not exists')
        spark.sql(f'CREATE DATABASE IF NOT EXISTS {database_name};')
        spark.sql(f'USE {database_name};')

    def run_data_ingest(self) -> pyspark.sql.DataFrame:
        """
        Run data ingest step

        Returns
        -------
        pyspark.sql.DataFrame
            Input Spark DataFrame
        """
        return spark.table(self.cfg.input_table)

    def run_data_prep(self, input_df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """
        Run data preparation step, using Featurizer to run featurization logic to create features from the input
        DataFrame.

        Parameters
        ----------
        input_df : pyspark.sql.DataFrame
            Input Spark DataFrame

        Returns
        -------
        pyspark.sql.DataFrame
            Processed Spark DataFrame containing features
        """
        featurizer = Featurizer(self.cfg.featurizer_cfg)
        proc_df = featurizer.run(input_df)

        return proc_df

    def run_feature_table_refresh(self, df: pyspark.sql.DataFrame) -> None:
        """
        Method to refresh feature table in Databricks Feature Store. When run, this method will refresh the feature
        table. As such, we first create (if it doesn't exist) the database specified, and drop the table if it already
        exists.

        The feature table is created from the Spark DataFrame provided, dropping the label column if it exists in the
        DataFrame. The label column cannot be present in the feature table when later constructing a feature store
        training set from the feature table. The feature table will be created using the primary keys and description
        provided via feature_store_table_cfg.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Spark DataFrame from which to create the feature table.
        """
        feature_store_table_cfg = self.cfg.feature_store_table_cfg

        # Create database and the table if not exists
        self.setup(database_name=feature_store_table_cfg.database_name)

        # Store only features for each customerID, storing customerID, churn in separate churn_labels table
        # During model training we will use the churn_labels table to join features into
        features_df = df.drop(self.cfg.labels_table_cfg.label_col)
        feature_table_name = feature_store_table_cfg.table_name
        database_name = feature_store_table_cfg.database_name
        _logger.info(f'Writing features to feature table: {feature_table_name}')
        feature_store_utils.create_and_write_feature_table(features_df,
                                                           feature_table_name,
                                                           database_name,
                                                           primary_keys=feature_store_table_cfg.primary_keys,
                                                           description=feature_store_table_cfg.description)

    def run_labels_table_refresh(self, df: pyspark.sql.DataFrame) -> None:
        """
        Method to refresh labels table. This table will consist of the columns primary_key, label_col

        Create table using params specified in labels_table_cfg. Will create Delta table at dbfs_path, and further
        create a table using this Delta location.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Spark DataFrame containing primary keys column and label column
        """
        feature_store_table_cfg = self.cfg.feature_store_table_cfg
        labels_table_cfg = self.cfg.labels_table_cfg

        if isinstance(feature_store_table_cfg.primary_keys, str):
            labels_table_cols = [feature_store_table_cfg.primary_keys,
                                 labels_table_cfg.label_col]
        elif isinstance(feature_store_table_cfg.primary_keys, list):
            labels_table_cols = feature_store_table_cfg.primary_keys + \
                                [labels_table_cfg.label_col]
        else:
            raise RuntimeError('Feature Store table primary keys must be one of either str of list type')

        labels_database_name = labels_table_cfg.database_name
        labels_table_name = labels_table_cfg.table_name
        labels_dbfs_path = labels_table_cfg.dbfs_path
        # Create database if not exists
        self.setup(database_name=labels_database_name)
        # Drop table if it exists
        spark.sql(f"""DROP TABLE IF EXISTS {labels_database_name}.{labels_table_name};""")
        # DataFrame of customerID/churn labels
        labels_df = df.select(labels_table_cols)
        _logger.info(f'Writing labels to DBFS: {labels_dbfs_path}')
        labels_df.write.format('delta').mode('overwrite').save(labels_dbfs_path)
        spark.sql(f"""CREATE TABLE {labels_database_name}.{labels_table_name}
                      USING DELTA LOCATION '{labels_dbfs_path}';""")
        _logger.info(f'Created labels table: {labels_database_name}.{labels_table_name}')

    def run(self) -> None:
        """
        Run feature table creation pipeline
        """
        _logger.info('==========Data Ingest==========')
        input_df = self.run_data_ingest()

        _logger.info('==========Data Prep==========')
        proc_df = self.run_data_prep(input_df)

        _logger.info('==========Create Feature Table==========')
        self.run_feature_table_refresh(proc_df)

        _logger.info('==========Create Labels Table==========')
        self.run_labels_table_refresh(proc_df)
