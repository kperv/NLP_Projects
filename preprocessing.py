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

def get_nclass_df(data_df, n_classes=n_classes):
    """Choose n_classes number of countries sorted by number
     and create a new dataframe with chosen countries.

    :param data_df: dataframe with reviews
    :param n_classes: should be 2 or more
    :return: a new smaller dataframe with n_classes unique countries
    """
    n_country_reviews = pd.DataFrame(columns=['country', 'description'])
    top_countries = data_df.country.value_counts()[:n_classes].keys()
    for country in top_countries:
        country_reviews = pd.DataFrame(data_df[data_df.country == country])
        n_country_reviews = n_country_reviews.append(country_reviews)
    return n_country_reviews

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