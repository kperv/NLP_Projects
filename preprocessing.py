"""This module consists of two classes to preprocess the row dataframe
to be ready for machine learning tasks.
Methods:
    get_nclass_df: pulls n classes out of the original dataframe
        to a new dataframe
    add_splits: calculates the number of rows to be used
        for training and validation according to attributes and creates
    a new dataframe with additional column, holding a string with a split
    name.
 """

__all__ = ["get_nclass_df", "add_splits"]


import pandas as pd


def get_nclass_df(data_df, n_classes=3):
    """Choose n_classes number of countries sorted by number
     and create a new dataframe with chosen countries.

    :param data_df: dataframe with reviews
    :param n_classes: should be 2 (for binary classeification)
                      or more (for multiclass classification)
    :return: a new smaller dataframe with n_classes unique countries

    >>> dummy_df = pd.DataFrame({})
    >>> n_class_df = get_nclass_df(dummy_df)
    DataFrame is empty
    >>> df = pd.DataFrame({
    ...     'review': ['Good wine', 'Bad wine', 'Woody wine'],
    ...     'country': ['US', 'France', 'Italy']
    ...     }
    ... )
    >>> n_class_df = get_nclass_df(df)
    >>> len(n_class_df.country.unique()) == n_classes
    True
    >>> n_class_df = get_nclass_df(df, 5)
    The maximum number of classes from this DataFrame is 3.
    >>> len(n_class_df.country.unique())
    3
    """
    num_too_big_msg = ("The maximum number of classes "
                       "from this DataFrame is {}.")
    try:
        original_num_classes = len(data_df.country.unique())
        if original_num_classes < n_classes:
            print(num_too_big_msg.format(original_num_classes))
            return data_df
        elif original_num_classes == n_classes:
            return data_df
        else:
            top_countries = data_df.country.value_counts()[:n_classes].keys()
            n_country_reviews = pd.DataFrame(
                columns=['country', 'description']
            )
            for country in top_countries:
                country_reviews = pd.DataFrame(
                    data_df[data_df.country == country]
                )
                n_country_reviews = n_country_reviews.append(country_reviews)
            return n_country_reviews
    except AttributeError:
        print("DataFrame is empty")
        return None

def add_splits(data_df, train_split=0.7, val_split=0.15):
    """Create a new dataframe with additional column 'split',
    calculate number of rows for each split
    and populate with a corresponding string the new column.

    :param data_df: a dataframe with reviews
    :param train_split: (float) proportion of dataframe
                        to be used for training
    :param val_split: the same for validation
    :return: a sorted dataframe with three columns
    """
    assert sum(train_split, val_split) <= 0.9  # a test set should be
                                               # at least 10%
    split_reviews = pd.DataFrame(columns=['country', 'description', 'split'])
    for country in data_df.country.unique():
        country_reviews = pd.DataFrame(data_df[data_df.country == country])
        n_total = len(country_reviews)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)

        country_reviews['split'] = None
        country_reviews.split.iloc[:n_train] = 'train'
        country_reviews.split.iloc[n_train:n_train+n_val] = 'val'
        country_reviews.split.iloc[n_train+n_val:] = 'test'

        split_reviews = split_reviews.append(country_reviews)
    return split_reviews