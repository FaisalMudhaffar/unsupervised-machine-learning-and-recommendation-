"""
project_tests.py
Udacity – Recommendations with IBM
Built-in test suite called from the main notebook.
"""

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Part I
# ──────────────────────────────────────────────────────────────────────────
def sol_1_test(sol_1_dict):
    """Validate Part I solution dictionary."""
    required = {
        '`50% of users interact with _____ number of articles or fewer.`',
        '`The total number of user-article interactions in the dataset is ______.`',
        '`The maximum number of articles that the same user has interacted with is ______.`',
        '`The maximum number of times an article has been viewed is ______.`',
        '`The most viewed article in the dataset as a string is ______.`',
        '`The number of unique articles that have at least one interaction are ______.`',
        '`The number of unique users in the dataset is ______`',
        '`The number of unique articles on the IBM platform`',
    }

    print("=" * 60)
    print("Part I – EDA Solution Test")
    print("=" * 60)
    passed = True

    for key in required:
        if key not in sol_1_dict:
            print(f"  ✗ MISSING KEY: {key}")
            passed = False
            continue
        val = sol_1_dict[key]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            print(f"  ✗ None/NaN for: {key}")
            passed = False
        else:
            label = key.strip('`').strip()
            print(f"  ✓ {label}: {val}")

    print()
    if passed:
        print("✅  All Part I checks PASSED!")
    else:
        print("❌  Some Part I checks FAILED – review above.")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────
# Part II
# ──────────────────────────────────────────────────────────────────────────
def ranked_recs_test(get_top_articles, get_top_article_ids, df):
    """Validate rank-based recommendation functions."""
    print("=" * 60)
    print("Part II – Rank-Based Recommendations Test")
    print("=" * 60)
    passed = True

    for n in [5, 10, 20]:
        names = get_top_articles(n, df)
        if not isinstance(names, list):
            print(f"  ✗ get_top_articles({n}) must return list, got {type(names)}")
            passed = False
        elif len(names) != n:
            print(f"  ✗ get_top_articles({n}) returned {len(names)} items (expected {n})")
            passed = False
        else:
            print(f"  ✓ get_top_articles({n}) → {n} titles")

        ids = get_top_article_ids(n, df)
        if not isinstance(ids, list):
            print(f"  ✗ get_top_article_ids({n}) must return list, got {type(ids)}")
            passed = False
        elif len(ids) != n:
            print(f"  ✗ get_top_article_ids({n}) returned {len(ids)} items (expected {n})")
            passed = False
        elif not all(isinstance(x, str) for x in ids):
            print(f"  ✗ get_top_article_ids({n}) items must be strings")
            passed = False
        else:
            print(f"  ✓ get_top_article_ids({n}) → {n} string ids")

    print()
    print("✅  All Part II checks PASSED!" if passed else "❌  Some Part II checks FAILED.")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────
# Part III
# ──────────────────────────────────────────────────────────────────────────
def user_item_matrix_test(user_item):
    """Validate the user-item binary matrix."""
    print("=" * 60)
    print("Part III – User-Item Matrix Test")
    print("=" * 60)
    passed = True

    if not isinstance(user_item, pd.DataFrame):
        print(f"  ✗ user_item must be a DataFrame, got {type(user_item)}")
        passed = False
    else:
        print(f"  ✓ DataFrame with shape {user_item.shape}")

    unique_vals = set(np.unique(user_item.values))
    extra = unique_vals - {0, 1}
    if extra:
        print(f"  ✗ Matrix contains values other than 0/1: {extra}")
        passed = False
    else:
        print("  ✓ Contains only 0s and 1s")

    null_count = user_item.isnull().sum().sum()
    if null_count > 0:
        print(f"  ✗ Matrix contains {null_count} NaN values")
        passed = False
    else:
        print("  ✓ No NaN values")

    print()
    print("✅  All Part III matrix checks PASSED!" if passed else "❌  Some checks FAILED.")
    print("=" * 60)


def cf_tests(user_user_recs, user_user_recs_part2, user_item, df):
    """Validate collaborative filtering recommendation functions."""
    print("=" * 60)
    print("Part III – Collaborative Filtering Tests")
    print("=" * 60)
    passed = True

    test_users = [user_item.index[0], user_item.index[1], user_item.index[2]]

    for uid in test_users:
        recs = user_user_recs(uid, m=10)
        if not isinstance(recs, list):
            print(f"  ✗ user_user_recs('{uid}') must return list")
            passed = False
        elif len(recs) == 0:
            print(f"  ✗ user_user_recs('{uid}') returned empty list")
            passed = False
        else:
            print(f"  ✓ user_user_recs('{uid}') → {len(recs)} recs")

        recs2, names2 = user_user_recs_part2(uid, m=10)
        if not isinstance(recs2, list) or not isinstance(names2, list):
            print(f"  ✗ user_user_recs_part2('{uid}') must return (list, list)")
            passed = False
        elif len(recs2) == 0:
            print(f"  ✗ user_user_recs_part2('{uid}') returned empty list")
            passed = False
        else:
            print(f"  ✓ user_user_recs_part2('{uid}') → {len(recs2)} recs")

    print()
    print("✅  All Part III CF checks PASSED!" if passed else "❌  Some checks FAILED.")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────
# Part V
# ──────────────────────────────────────────────────────────────────────────
def svd_test(U, sigma, Vt, user_item_matrix):
    """Validate SVD decomposition matrices."""
    print("=" * 60)
    print("Part V – SVD Test")
    print("=" * 60)
    passed = True

    n_users, n_articles = user_item_matrix.shape
    k = len(sigma)

    if U.shape[0] != n_users:
        print(f"  ✗ U rows should be {n_users}, got {U.shape[0]}")
        passed = False
    else:
        print(f"  ✓ U shape: {U.shape}")

    print(f"  ✓ Sigma shape: ({k},)")

    if Vt.shape[1] != n_articles:
        print(f"  ✗ Vt cols should be {n_articles}, got {Vt.shape[1]}")
        passed = False
    else:
        print(f"  ✓ Vt shape: {Vt.shape}")

    # Full reconstruction error
    reconstructed = np.dot(U * sigma, Vt)
    mse = np.mean((reconstructed - user_item_matrix) ** 2)
    if mse < 1e-6:
        print(f"  ✓ Full reconstruction MSE: {mse:.2e}  (near-perfect ✓)")
    else:
        print(f"  ⚠ Full reconstruction MSE: {mse:.4f}  (higher than expected)")

    print()
    print("✅  All Part V SVD checks PASSED!" if passed else "❌  Some checks FAILED.")
    print("=" * 60)


if __name__ == "__main__":
    print("project_tests.py ready. Available tests:")
    for fn in [sol_1_test, ranked_recs_test, user_item_matrix_test, cf_tests, svd_test]:
        print(f"  • {fn.__name__}")
