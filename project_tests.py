"""
project_tests.py
Udacity – Recommendations with IBM
Test suite to verify correctness of solution functions.
"""

import pandas as pd
import numpy as np


def sol_1_test(sol_1_dict):
    """
    Validates the solution dictionary for Part I (EDA).
    Checks that required keys are present and values are of the correct type.
    """
    required_keys = [
        '`50% of users interact with _____ number of articles or fewer.`',
        '`The total number of user-article interactions in the dataset is ______.`',
        '`The maximum number of articles that the same user has interacted with is ______.`',
        '`The maximum number of times an article has been viewed is ______.`',
        '`The most viewed article in the dataset as a string is ______.`',
        '`The number of unique articles that have at least one interaction are ______.`',
        '`The number of unique users in the dataset is ______`',
        '`The number of unique articles on the IBM platform`'
    ]

    print("=" * 60)
    print("Part I – Solution Test")
    print("=" * 60)
    passed = True

    for key in required_keys:
        if key not in sol_1_dict:
            print(f"  ✗ MISSING KEY: {key}")
            passed = False
        else:
            val = sol_1_dict[key]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                print(f"  ✗ NONE/NaN value for: {key}")
                passed = False
            else:
                print(f"  ✓ {key.strip('`')}: {val}")

    print()
    if passed:
        print("✅ All Part I checks passed!")
    else:
        print("❌ Some Part I checks failed. Please review.")
    print("=" * 60)


def ranked_recs_test(get_top_articles, get_top_article_ids, df):
    """
    Tests rank-based recommendation functions.
    """
    print("=" * 60)
    print("Part II – Rank-Based Recommendations Test")
    print("=" * 60)

    passed = True

    # Test get_top_articles
    for n in [5, 10, 20]:
        result = get_top_articles(n, df)
        if not isinstance(result, list):
            print(f"  ✗ get_top_articles({n}) should return a list, got {type(result)}")
            passed = False
        elif len(result) != n:
            print(f"  ✗ get_top_articles({n}) returned {len(result)} items, expected {n}")
            passed = False
        else:
            print(f"  ✓ get_top_articles({n}) returned {n} items")

    # Test get_top_article_ids
    for n in [5, 10, 20]:
        result = get_top_article_ids(n, df)
        if not isinstance(result, list):
            print(f"  ✗ get_top_article_ids({n}) should return a list, got {type(result)}")
            passed = False
        elif len(result) != n:
            print(f"  ✗ get_top_article_ids({n}) returned {len(result)} items, expected {n}")
            passed = False
        elif not all(isinstance(x, str) for x in result):
            print(f"  ✗ get_top_article_ids({n}) should return strings")
            passed = False
        else:
            print(f"  ✓ get_top_article_ids({n}) returned {n} string ids")

    print()
    if passed:
        print("✅ All Part II checks passed!")
    else:
        print("❌ Some Part II checks failed. Please review.")
    print("=" * 60)


def user_item_matrix_test(user_item):
    """
    Tests the user-item matrix.
    """
    print("=" * 60)
    print("Part III – User-Item Matrix Test")
    print("=" * 60)

    passed = True

    if not isinstance(user_item, pd.DataFrame):
        print("  ✗ user_item should be a pandas DataFrame")
        passed = False
    else:
        print(f"  ✓ user_item is a DataFrame with shape {user_item.shape}")

    unique_vals = set(user_item.values.flatten())
    if unique_vals - {0, 1}:
        print(f"  ✗ user_item should contain only 0s and 1s, found: {unique_vals - {0, 1}}")
        passed = False
    else:
        print("  ✓ user_item contains only 0s and 1s")

    print()
    if passed:
        print("✅ All Part III matrix checks passed!")
    else:
        print("❌ Some Part III checks failed. Please review.")
    print("=" * 60)


def svd_test(U, sigma, Vt, user_item_matrix):
    """
    Tests the SVD decomposition.
    """
    print("=" * 60)
    print("Part V – SVD Test")
    print("=" * 60)

    passed = True

    # Check shapes
    n_users, n_articles = user_item_matrix.shape
    k = len(sigma)

    if U.shape[0] != n_users:
        print(f"  ✗ U should have {n_users} rows (users), got {U.shape[0]}")
        passed = False
    else:
        print(f"  ✓ U shape: {U.shape}")

    if Vt.shape[1] != n_articles:
        print(f"  ✗ Vt should have {n_articles} columns (articles), got {Vt.shape[1]}")
        passed = False
    else:
        print(f"  ✓ Vt shape: {Vt.shape}")

    print(f"  ✓ Sigma shape: ({k},)")

    # Check reconstruction
    reconstructed = np.dot(np.dot(U, np.diag(sigma)), Vt)
    reconstruction_error = np.mean((reconstructed - user_item_matrix) ** 2)
    print(f"  ✓ Full reconstruction MSE: {reconstruction_error:.6f}")

    print()
    if passed:
        print("✅ All Part V SVD checks passed!")
    else:
        print("❌ Some Part V checks failed. Please review.")
    print("=" * 60)


if __name__ == "__main__":
    print("project_tests.py loaded successfully.")
    print("Available tests:")
    print("  - sol_1_test(sol_1_dict)")
    print("  - ranked_recs_test(get_top_articles, get_top_article_ids, df)")
    print("  - user_item_matrix_test(user_item)")
    print("  - svd_test(U, sigma, Vt, user_item_matrix)")
