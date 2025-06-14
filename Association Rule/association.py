import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth, association_rules
from fpgrowth_py import fpgrowth as fpgrowth_py
from efficient_apriori import apriori as efficient_apriori
from fim import apriori as fim_apriori, fpgrowth as fim_fpgrowth, eclat
from collections import defaultdict

# ==============================================
# Data Preparation
# ==============================================

# Sample transaction data (replace with your actual data)
transactions = [
    ['milk', 'bread', 'eggs'],
    ['milk', 'bread', 'diapers', 'beer'],
    ['bread', 'eggs', 'diapers'],
    ['milk', 'eggs', 'diapers', 'beer'],
    ['bread', 'milk', 'eggs', 'diapers'],
    ['bread', 'milk', 'diapers', 'beer']
]

# Convert to one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)


# ==============================================
# Define ALL Association Rule Algorithms
# ==============================================

def run_apriori(df, min_support=0.5, min_confidence=0.7):
    """Standard Apriori implementation from mlxtend"""
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return frequent_itemsets, rules


def run_fpgrowth(df, min_support=0.5, min_confidence=0.7):
    """FP-Growth implementation from mlxtend"""
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return frequent_itemsets, rules


def run_fpmax(df, min_support=0.5, min_confidence=0.7):
    """FP-Max implementation from mlxtend"""
    frequent_itemsets = fpmax(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return frequent_itemsets, rules


def run_efficient_apriori(transactions, min_support=0.5, min_confidence=0.7):
    """More memory-efficient Apriori implementation"""
    itemsets, rules = efficient_apriori(transactions,
                                        min_support=min_support,
                                        min_confidence=min_confidence)
    return itemsets, rules


def run_fpgrowth_py(transactions, min_support=0.5, min_confidence=0.7):
    """Alternative FP-Growth implementation"""
    freq_itemsets, rules = fpgrowth_py(transactions, minSupRatio=min_support, minConf=min_confidence)
    return freq_itemsets, rules


def run_fim_apriori(transactions, min_support=0.5, min_confidence=0.7):
    """Apriori from FIM (Frequent Itemset Mining) package"""
    itemsets = fim_apriori(transactions, target='r', supp=min_support * 100, conf=min_confidence * 100)
    return itemsets, []


def run_fim_fpgrowth(transactions, min_support=0.5, min_confidence=0.7):
    """FP-Growth from FIM package"""
    itemsets = fim_fpgrowth(transactions, target='r', supp=min_support * 100, conf=min_confidence * 100)
    return itemsets, []


def run_eclat(transactions, min_support=0.5):
    """ECLAT algorithm implementation"""
    itemsets = eclat(transactions, target='s', supp=min_support * 100)
    return itemsets, []


def run_relim(transactions, min_support=0.5):
    """RElim algorithm implementation (recursive elimination)"""
    # This is a simplified implementation
    items = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            items[item] += 1

    n_transactions = len(transactions)
    freq_items = {item: count / n_transactions
                  for item, count in items.items()
                  if count / n_transactions >= min_support}

    return freq_items, []


# ==============================================
# Evaluation Function
# ==============================================

def evaluate_association_rules(rules, algorithm_name):
    if len(rules) == 0:
        print(f"\nNo rules found for {algorithm_name}")
        return None

    metrics = {
        'num_rules': len(rules),
        'avg_support': rules['support'].mean(),
        'avg_confidence': rules['confidence'].mean(),
        'avg_lift': rules['lift'].mean(),
        'avg_conviction': rules['conviction'].mean() if 'conviction' in rules.columns else None,
        'avg_leverage': rules['leverage'].mean() if 'leverage' in rules.columns else None
    }

    print(f"\nAssociation Rules Metrics for {algorithm_name}:")
    for name, value in metrics.items():
        if value is not None:
            print(f"{name}: {value:.4f}")

    return metrics


# ==============================================
# Algorithm Training and Evaluation
# ==============================================

results = {}
all_rules = {}

# Define parameter grid
param_grid = {
    'min_support': [0.3, 0.4, 0.5],
    'min_confidence': [0.5, 0.6, 0.7]
}

# Run all algorithms with different parameter combinations
for min_support in param_grid['min_support']:
    for min_confidence in param_grid['min_confidence']:
        print(f"\n{'=' * 50}")
        print(f"Testing with min_support={min_support}, min_confidence={min_confidence}")
        print(f"{'=' * 50}")

        # 1. Apriori
        try:
            apriori_itemsets, apriori_rules = run_apriori(df, min_support, min_confidence)
            metrics = evaluate_association_rules(apriori_rules, f"Apriori (sup={min_support}, conf={min_confidence})")
            if metrics:
                results[f"Apriori_sup{min_support}_conf{min_confidence}"] = metrics
                all_rules[f"Apriori_sup{min_support}_conf{min_confidence}"] = apriori_rules
        except Exception as e:
            print(f"Error with Apriori: {str(e)}")

        # 2. FP-Growth
        try:
            fpgrowth_itemsets, fpgrowth_rules = run_fpgrowth(df, min_support, min_confidence)
            metrics = evaluate_association_rules(fpgrowth_rules,
                                                 f"FP-Growth (sup={min_support}, conf={min_confidence})")
            if metrics:
                results[f"FP-Growth_sup{min_support}_conf{min_confidence}"] = metrics
                all_rules[f"FP-Growth_sup{min_support}_conf{min_confidence}"] = fpgrowth_rules
        except Exception as e:
            print(f"Error with FP-Growth: {str(e)}")

        # 3. FP-Max
        try:
            fpmax_itemsets, fpmax_rules = run_fpmax(df, min_support, min_confidence)
            metrics = evaluate_association_rules(fpmax_rules, f"FP-Max (sup={min_support}, conf={min_confidence})")
            if metrics:
                results[f"FP-Max_sup{min_support}_conf{min_confidence}"] = metrics
                all_rules[f"FP-Max_sup{min_support}_conf{min_confidence}"] = fpmax_rules
        except Exception as e:
            print(f"Error with FP-Max: {str(e)}")

        # 4. Efficient Apriori
        try:
            eff_apriori_itemsets, eff_apriori_rules = run_efficient_apriori(transactions, min_support, min_confidence)
            # Convert to DataFrame for consistent evaluation
            rules_list = []
            for rule in eff_apriori_rules:
                rules_list.append({
                    'antecedents': frozenset(rule.lhs),
                    'consequents': frozenset(rule.rhs),
                    'support': rule.support,
                    'confidence': rule.confidence,
                    'lift': rule.lift
                })
            eff_apriori_rules_df = pd.DataFrame(rules_list)
            metrics = evaluate_association_rules(eff_apriori_rules_df,
                                                 f"Efficient Apriori (sup={min_support}, conf={min_confidence})")
            if metrics:
                results[f"Efficient Apriori_sup{min_support}_conf{min_confidence}"] = metrics
                all_rules[f"Efficient Apriori_sup{min_support}_conf{min_confidence}"] = eff_apriori_rules_df
        except Exception as e:
            print(f"Error with Efficient Apriori: {str(e)}")

        # 5. FP-Growth (alternative implementation)
        try:
            fpgrowth_py_itemsets, fpgrowth_py_rules = run_fpgrowth_py(transactions, min_support, min_confidence)
            # Convert to DataFrame for consistent evaluation
            rules_list = []
            for rule in fpgrowth_py_rules:
                rules_list.append({
                    'antecedents': frozenset(rule[0]),
                    'consequents': frozenset(rule[1]),
                    'support': rule[2],
                    'confidence': rule[3],
                    'lift': rule[4]
                })
            fpgrowth_py_rules_df = pd.DataFrame(rules_list)
            metrics = evaluate_association_rules(fpgrowth_py_rules_df,
                                                 f"FP-Growth-py (sup={min_support}, conf={min_confidence})")
            if metrics:
                results[f"FP-Growth-py_sup{min_support}_conf{min_confidence}"] = metrics
                all_rules[f"FP-Growth-py_sup{min_support}_conf{min_confidence}"] = fpgrowth_py_rules_df
        except Exception as e:
            print(f"Error with FP-Growth-py: {str(e)}")

        # 6. FIM Apriori
        try:
            fim_apriori_itemsets, _ = run_fim_apriori(transactions, min_support, min_confidence)
            # Convert itemsets to rules format for evaluation
            rules_list = []
            for rule in fim_apriori_itemsets:
                lhs, rhs, support, confidence = rule
                rules_list.append({
                    'antecedents': frozenset(lhs),
                    'consequents': frozenset(rhs),
                    'support': support / 100,
                    'confidence': confidence / 100,
                    'lift': None  # FIM doesn't calculate lift
                })
            fim_apriori_rules_df = pd.DataFrame(rules_list)
            metrics = evaluate_association_rules(fim_apriori_rules_df,
                                                 f"FIM Apriori (sup={min_support}, conf={min_confidence})")
            if metrics:
                results[f"FIM Apriori_sup{min_support}_conf{min_confidence}"] = metrics
                all_rules[f"FIM Apriori_sup{min_support}_conf{min_confidence}"] = fim_apriori_rules_df
        except Exception as e:
            print(f"Error with FIM Apriori: {str(e)}")

        # 7. FIM FP-Growth
        try:
            fim_fpgrowth_itemsets, _ = run_fim_fpgrowth(transactions, min_support, min_confidence)
            # Convert itemsets to rules format for evaluation
            rules_list = []
            for rule in fim_fpgrowth_itemsets:
                lhs, rhs, support, confidence = rule
                rules_list.append({
                    'antecedents': frozenset(lhs),
                    'consequents': frozenset(rhs),
                    'support': support / 100,
                    'confidence': confidence / 100,
                    'lift': None  # FIM doesn't calculate lift
                })
            fim_fpgrowth_rules_df = pd.DataFrame(rules_list)
            metrics = evaluate_association_rules(fim_fpgrowth_rules_df,
                                                 f"FIM FP-Growth (sup={min_support}, conf={min_confidence})")
            if metrics:
                results[f"FIM FP-Growth_sup{min_support}_conf{min_confidence}"] = metrics
                all_rules[f"FIM FP-Growth_sup{min_support}_conf{min_confidence}"] = fim_fpgrowth_rules_df
        except Exception as e:
            print(f"Error with FIM FP-Growth: {str(e)}")

        # 8. ECLAT
        try:
            eclat_itemsets, _ = run_eclat(transactions, min_support)
            # ECLAT only finds frequent itemsets, not rules
            metrics = {
                'num_itemsets': len(eclat_itemsets),
                'avg_support': sum(sup for _, sup in eclat_itemsets) / len(eclat_itemsets) if eclat_itemsets else 0
            }
            print(f"\nECLAT Metrics (sup={min_support}):")
            for name, value in metrics.items():
                print(f"{name}: {value:.4f}")
            results[f"ECLAT_sup{min_support}"] = metrics
        except Exception as e:
            print(f"Error with ECLAT: {str(e)}")

        # 9. RElim
        try:
            relim_itemsets, _ = run_relim(transactions, min_support)
            # RElim only finds frequent items, not rules
            metrics = {
                'num_items': len(relim_itemsets),
                'avg_support': sum(relim_itemsets.values()) / len(relim_itemsets) if relim_itemsets else 0
            }
            print(f"\nRElim Metrics (sup={min_support}):")
            for name, value in metrics.items():
                print(f"{name}: {value:.4f}")
            results[f"RElim_sup{min_support}"] = metrics
        except Exception as e:
            print(f"Error with RElim: {str(e)}")

# ==============================================
# Model Comparison
# ==============================================

print("\nAlgorithm Comparison:")
comparison = pd.DataFrame.from_dict(results, orient='index')
print(comparison.sort_values(by='num_rules' if 'num_rules' in comparison.columns else 'num_itemsets', ascending=False))

# ==============================================
# Save Best Rules
# ==============================================

# Find the algorithm with most rules (or highest avg lift if available)
if 'avg_lift' in comparison.columns:
    best_algorithm = comparison['avg_lift'].idxmax()
else:
    best_algorithm = comparison['num_rules'].idxmax() if 'num_rules' in comparison.columns else comparison.index[0]

print(f"\nBest algorithm is: {best_algorithm}")

if best_algorithm in all_rules:
    best_rules = all_rules[best_algorithm]
    print("\nTop 5 rules from best algorithm:")
    print(best_rules.head().to_string())

    # Save the best rules to CSV
    best_rules.to_csv('best_association_rules.csv', index=False)
else:
    print("\nBest algorithm doesn't produce rules (itemset mining only)")

print("\nAssociation rule mining complete!")