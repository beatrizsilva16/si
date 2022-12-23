from scipy.stats import stats

from si.data.dataset import Dataset

def f_classification(dataset: Dataset):
    """
    Calculates the score for each feature for classification tasks.
    :param dataset: Dataset object.
    :return: F value for each feature.
    """
    classes_dt = dataset.get_classes()
    groups_dt = [dataset.x[dataset.y == c] for c in classes_dt]  # group the dataset by class
    f_value, p_value = stats.f_oneway(*groups_dt)

    return f_value, p_value
