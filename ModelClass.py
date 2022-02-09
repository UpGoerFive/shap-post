import pandas as pd
import numpy as np
import seaborn as sns
import logging
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
# Current version of sklearn is still too old I think, might try to upgrade to use below option
# from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV# ,  HalvingRandomSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector
import sklearn.metrics as metrics
from sklearn.inspection import permutation_importance

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('model-run.log')
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
# logger.addHandler(c_handler)
logger.addHandler(f_handler)
logger.info("Logger set up")


class Modeler:
    """
    Modeling pipeline. It has basic defaults and can accept new models and transformers.
    Models should be added in the form of:

    {'classifier': <classifier>,
     'preprocessor': <preprocessor>}

    preprocessor can be None if the default preprocessor is acceptable. This class also
    logs model output to a default model-run.log file. Each train or test method also has an optional print
    keyword argument that will print output if desired, as well as log it to the output file. This defaults
    to True for single runs, and False for multiple runs.
    """
    def __init__(self, models = None, X=pd.DataFrame(), y=pd.DataFrame()):
        if models is None:
            models = {}
        self._models = {}
        self._tuning = {}

        if X.empty or y.empty:
            raise Exception('X and y should be provided at start.')

        self._le = LabelEncoder()
        self._X_train, self._X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 829941045)
        self._y_train = self._le.fit_transform(y_train.iloc[:, 1])
        self._y_test = self._le.transform(y_test.iloc[:, 1])
        for key, value in models.items():
                    self.add_model(key, value)

    def create_default_prep(self, cat_add=None, num_add=None):
        """
        Creates a default preprocessing object, uses all columns and imputes with median for numeric and 'missing' for categorical.
        Can accept extra steps with cat_add and num_add, which must be lists of tuples (steps). Currently only adds them to the order.
        """

        def to_object(x):
            return pd.DataFrame(x).astype(str)

        string_transformer = FunctionTransformer(to_object)

        if num_add:
            numeric_transformer = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='median')),
                        *num_add]
            )
        else:
            numeric_transformer = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='median'))]
            )

        if cat_add:
            categorical_transformer = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                    ('casting', string_transformer),
                    ('one_hot_encode', OneHotEncoder(handle_unknown='ignore')),
                    *cat_add]
            )
        else:
            categorical_transformer = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                    ('casting', string_transformer),
                    ('one_hot_encode', OneHotEncoder(handle_unknown='ignore'))]
            )

        preprocessor = ColumnTransformer(
                                transformers=[
                                    ("numeric", numeric_transformer, make_column_selector(dtype_include=np.number)),
                                    ("categorical", categorical_transformer, make_column_selector(dtype_exclude=np.number))])

        return preprocessor

    def add_model(self, name, model):
        """
        Basic mechanism to add a model, model must provide a classifier field, and can optionally provide a preprocessor field.
        Model can have None as the preprocessor, in which case a default will be provided.
        """
        if 'preprocessor' not in model.keys():
                preprocessor = self.create_default_prep()
                model['preprocessor'] = preprocessor
        else:
            preprocessor = model['preprocessor']

        self._models[name] = model

        if 'model_pipeline' not in model.keys():
            self._models[name]['model_pipeline'] = Pipeline(steps=[('preprocessor', preprocessor),
                                                                    ('classifier', self._models[name]['classifier'])])

    def remove_model(self, name):
        """
        Files your taxes.
        """
        del self._models[name]

    def change_prep(self, name, prep):
        """
        Basic reassignment of preprocessor pipeline object.
        """
        self._models[name]['preprocessor'] = prep

    def show_model(self, name):
        """
        Shows all model information.
        To Do: add printing/logging options.
        """
        print(f"{name}: {self._models[name]}")

    def get_model(self, name):
        """
        Access a model to use.
        """
        return self._models[name]

    def train_model(self, name, print=True, cv=True, train=True):
        """
        Train a single model. Fits all preprocessing transformers for later testing.
        Records and outputs cross validate scores. The cv_only option determines if the method will
        fit a classifier, which is required before testing. Optional printing ability.
        """
        if print:
            logger.addHandler(c_handler)

        X_train = self._X_train
        y_train = self._y_train
        model = self._models[name]
        model_pipeline = model['model_pipeline']

        if cv:
            model['cv_output'] = cross_val_score(
                estimator=model_pipeline,
                X=X_train,
                y=y_train
            )
            logger.info(f"Cross validate scores for {name}: {model['cv_output']}")
            self._models[name]['time_cross_val'] = time.asctime()

        if train:
            model_pipeline = model_pipeline.fit(X_train, y_train)
            model['train_output'] = model_pipeline.score(X_train, y_train)
            logger.info(f"{name} has been fit.")
            self._models[name]['time_fit'] = time.asctime()

        if print:
            logger.removeHandler(c_handler)

    def train_all(self, print=False, cv=True, train=True):
        """
        Train all available models. Fits all preprocessing transformers for later testing.
        Records and outputs cross validate scores. The cv_only option determines if the method will
        fit a classifier, which is required before testing. Optional printing ability.
        """
        for model in self._models:
            self.train_model(model, print, cv, train)

    def test_model(self, name, print=True):
        """
        Test a single model. Uses already fitted preprocessor pipeline and classifier.
        Raises an exception if there is no fit classifier for the model. Optional printing.
        """
        if print:
            logger.addHandler(c_handler)

        X_test = self._X_test
        y_test = self._y_test
        model = self._models[name]
        model_pipeline = model['model_pipeline']

        model['test_output'] = model_pipeline.score(X_test, y_test)
        self._models[name]['time_tested'] = time.asctime()
        logger.info(f"{name} test score: {model['test_output']}")

        if print:
            logger.removeHandler(c_handler)

    def test_all(self, print=False):
        """
        Test all available models. Uses already fitted preprocessor pipelines and classifiers.
        Raises an exception if there is no fit classifier for a model. Optional printing.
        """

        for model in self._models:
            self.test_model(model, print)

    def hyper_search(self, name, searcher=RandomizedSearchCV, params=None, searcher_kwargs=None, print=False, set_to_train=False):
        """
        Hyper parameter tuning function, defaults to RandomizedSearchCV, but any search function
        you want can be passed in. searcher_kwargs should be a dictionary of the keyword argument you want to pass
        to the search object:

            searcher_kwargs = {'n_jobs': 3, 'refit': True, 'cv': 10}

        The keys need to be the exact arguments of the object. Note that this should not include things like
        param_distributions, as this should be filled in the params argument.
        """
        if print:
            logger.addHandler(c_handler)

        model = self._models[name]
        model_pipeline = model['model_pipeline']

        if not params and 'param_distro' in self._models[name].keys():
            params = self._models[name]['param_distro']
        elif params:
            params = {'classifier__' + key: value for key, value in params.items()}
            self._models[name]['param_distro'] = params

        if searcher_kwargs:
            search_object = searcher(model_pipeline, params, **searcher_kwargs)
        else:
            search_object = searcher(model_pipeline, params)

        search_object.fit(self._X_train, self._y_train)
        logger.info(f"For model {name}, {searcher.__name__} with{params} produced:")
        logger.info(f"Params: {search_object.best_params_}")
        logger.info(f"The mean cross validated score of the best estimator was :{search_object.best_score_}" if 'refit' not in searcher_kwargs.keys() else "refit = False")

        self._models[name]['search_classifier'] = search_object.best_estimator_ if 'refit' not in searcher_kwargs.keys() else None
        self._models[name]['search_best_params'] = search_object.best_params_
        self._models[name]['search_performed_at'] = time.asctime()

        if set_to_train:
            model['model_pipeline']= search_object.best_estimator_
            model['train_output'] = model['model_pipeline'].score(self._X_train, self._y_train)
            model['time_fit'] = time.asctime()

        if print:
            logger.removeHandler(c_handler)

    def model_evaluation(self, name, normalize="true", cmap="Purples", label=""):
        """
        Evaluates a classifier model by providing
            [1] Metrics including accuracy and cross validation score.
            [2] Classification report
            [3] Confusion Matrix
        Args:
            name (string): classifier model name
            normalize (str): "true" if normalize confusion matrix annotated values.
            cmap (str): color map for the confusion matrix
            label (str): name of the classifier.
        Returns:
            report: classfication report
            fig, ax: matplotlib object
        """

        # If the model hasn't been trained and tested yet, let's do that.
        if 'train_output' not in self._models[name].keys():
            self.train_model(name, cv=False, print=False)
            self.test_model(name, print=False)
        elif 'test_output' not in self._models[name].keys(): # No sense training if we don't have to.
            self.test_model(name, print=False)

        model = self._models[name]
        model_pipeline = model['model_pipeline']

        X_test = self._X_test
        y_test = self._y_test

        ## Get Predictions
        y_hat_test = model_pipeline.predict(X_test)

        ## Classification Report / Scores
        table_header = "[i] CLASSIFICATION REPORT"    ## Add Label if given
        if len(label)>0:
            table_header += f" {label}"
        ## PRINT CLASSIFICATION REPORT
        dashes = "---"*20
        print(dashes,table_header,dashes,sep="\n")
        print("Train Accuracy : ", round(self._models[name]['train_output'],4))
        print("Test Accuracy : ", round(self._models[name]['test_output'],4))

        if 'cv_output' in model.keys():
            print('CV score (n=5)', round(np.mean(self._models[name]['cv_output']), 4))
        print(dashes+"\n")

        y_label_test = self._le.inverse_transform(y_test)
        y_label_hat_test = self._le.inverse_transform(y_hat_test)
        print(metrics.classification_report(y_label_test, y_label_hat_test, target_names=self._le.classes_))
        model['report'] = metrics.classification_report(y_label_test, y_label_hat_test, target_names=self._le.classes_, output_dict=True)
        print(dashes+"\n\n")
        ## MAKE FIGURE
        fig, ax = plt.subplots(figsize=(10,4))
        ax.grid(False)
        ## Plot Confusion Matrix
        metrics.plot_confusion_matrix(model_pipeline, X_test,y_test,
                                    display_labels=self._le.classes_,
                                    normalize=normalize,
                                    cmap=cmap,ax=ax)
        ax.set(title="Confusion Matrix")
        plt.xticks(rotation=45)
        plt.show()
        return fig, ax

    def permutation_importance(self, name, train=False, perm_kwargs=None, save_graph=None):
        """
        Graphs and returns permutation importance of a model. Can be run on test or train data with the train
        option. If providing perm_kwargs, they should be in a dictionary of keys that correspond to the
        permutation importance function parameters.
        """
        model = self._models[name]
        model_pipeline = model['model_pipeline']

        X_val, y_val = (self._X_train, self._y_train) if train else (self._X_test, self._y_test)

        model_permuter = permutation_importance(model_pipeline, X_val, y_val, **perm_kwargs) if perm_kwargs else permutation_importance(model_pipeline, X_val, y_val)
        model['permuter'] = model_permuter

        # Plotting
        fig, ax = plt.subplots(figsize=(10,4))
        perm_imp = pd.Series(model_permuter.importances_mean, index=X_val.columns).sort_values(ascending=False)[:10]
        perm_imp.plot(kind="barh", title="Permutation Importances")
        ax.set(ylabel="Mean Permutation Importance Score")
        ax.invert_yaxis()

        if save_graph:
            plt.savefig(save_graph)
        logger.info(f"Model {name} has permutation importances of {perm_imp}")

    def plot_models(self, sns_style='darkgrid', sns_context='talk', palette='coolwarm', save=None, labels=None):
        """
        Skylar slide style, with thanks to Matt. Has options for seaborn plotting. If you want to save the plot,
        give the save option a filename, exactly as would be done with plt.savefig() Labels must be provided as a
        dictionary with the model names as keys and the Label you'd like to display as a value.
        """
        logger.removeHandler(c_handler)
        logger.removeHandler(f_handler)

        xticklabels = [labels[key] for key in self._models.keys()] if labels else list(self._models.keys())
        y = [model['test_output'] for model in self._models.values()]

        sns.set_style(sns_style)
        sns.set_context(sns_context)
        fig, ax = plt.subplots(figsize=(20, 10))

        fig.set_tight_layout(True)

        sns.barplot(x=xticklabels, y=y, palette=palette)
        ax.set(ylim=(0, 1))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

        ax.set_ylabel('Accuracy Score')
        ax.set_title('Model Effectiveness');

        if save:
            plt.savefig(save)

        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
