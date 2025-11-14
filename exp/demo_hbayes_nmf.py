"""
Demonstration of Hierarchical Bayesian NMF for Factor Count Determination
=========================================================================

This script demonstrates the different approaches for determining optimal
factor counts in source apportionment using Hierarchical Bayesian NMF.
"""

import numpy as np
from hbayes_nmf import HierarchicalBayesianNMF, compare_factor_selection_approaches

# Import the actual generate_dataset function from utils
from utils import generate_dataset, q_loss

# Global dataset variables to ensure consistency across approaches
SHARED_DATASET = None
SHARED_SPECIES_NAMES = None

def get_shared_dataset(true_k=5, n_samples=300, n_features=20, seed=42, force_regenerate=False):
    """Get or generate a shared dataset for all approaches to use"""
    global SHARED_DATASET, SHARED_SPECIES_NAMES

    if SHARED_DATASET is None or force_regenerate:
        print(f"Generating shared dataset with {true_k} true factors...")
        V, U, actual_k, true_H, true_W, simulator = generate_dataset(
            true_k=true_k,
            n_samples=n_samples,
            n_features=n_features,
            seed=seed
        )

        SHARED_SPECIES_NAMES = [f'Species_{i+1}' for i in range(V.shape[1])]

        SHARED_DATASET = {
            'V': V,
            'U': U,
            'true_k': actual_k,
            'true_H': true_H,
            'true_W': true_W,
            'simulator': simulator,
            'species_names': SHARED_SPECIES_NAMES
        }

        print(f"Dataset generated: {V.shape[0]} samples × {V.shape[1]} species")
        print(f"True number of factors: {actual_k}")
        print(f"Dataset statistics:")
        print(f"  - Concentration range: [{V.min():.3f}, {V.max():.3f}]")
        print(f"  - Mean uncertainty: {U.mean():.3f}")
        print(f"  - SNR (mean): {(V/U).mean():.1f}")

    return SHARED_DATASET


def demonstrate_single_approach(approach="ard", use_shared=True, true_k=5):
    """Demonstrate a single factor selection approach"""

    print(f"\n{'='*60}")
    print(f"DEMONSTRATING {approach.upper()} APPROACH")
    print(f"{'='*60}")

    if use_shared:
        # Use shared dataset for consistent comparison
        dataset = get_shared_dataset(true_k=true_k)
        V = dataset['V']
        U = dataset['U']
        true_k = dataset['true_k']
        species_names = dataset['species_names']
        true_H = dataset['true_H']
        true_W = dataset['true_W']
        print("Using shared dataset for consistent comparison")
    else:
        # Generate new dataset
        V, U, true_k, true_H, true_W, _ = generate_dataset(
            true_k=true_k, n_samples=300, n_features=20, seed=42
        )
        species_names = [f'Species_{i+1}' for i in range(V.shape[1])]

    print(f"Dataset: {V.shape[0]} samples × {V.shape[1]} species")
    print(f"True number of factors: {true_k}")

    # Initialize model
    model = HierarchicalBayesianNMF(
        max_factors=min(12, true_k + 5),  # Adaptive max based on true factors
        min_factors=2,
        approach=approach,
        uncertainty_estimation=True,
        seed=42
    )

    # Fit model
    print(f"\nFitting {approach} model...")
    model.fit(
        X=V,
        X_uncertainty=U,
        species_names=species_names,
        n_samples=1500,  # Reduced for demo
        n_tune=750,
        n_chains=4,
        target_accept=0.95
    )

    # Results
    print(f"\nRESULTS:")
    print(f"Optimal factors found: {model.optimal_factors}")
    print(f"True factors: {true_k}")
    accuracy = "✓ CORRECT" if model.optimal_factors == true_k else "✗ INCORRECT"
    print(f"Factor selection accuracy: {accuracy}")
    print(f"Model R²: {model.model_diagnostics['r2']:.3f}")
    print(f"RMSE: {model.model_diagnostics['rmse']:.3f}")

    # Calculate Q-loss for comparison with your existing workflow
    q_loss_value = q_loss(V, U, model.factor_contributions, model.factor_profiles)
    print(f"Q-loss: {q_loss_value:.2f}")

    # Plot results
    try:
        model.plot_factor_selection_results()
    except Exception as e:
        print(f"Plotting failed: {e}")

    # Print selection summary
    summary = model.get_model_selection_summary()
    print(f"\nSelection Summary:")
    for key, value in summary.items():
        if isinstance(value, (int, float, str)):
            print(f"  {key}: {value}")

    # Store results for comparison
    results = {
        'model': model,
        'optimal_factors': model.optimal_factors,
        'true_factors': true_k,
        'accuracy': model.optimal_factors == true_k,
        'r2': model.model_diagnostics['r2'],
        'rmse': model.model_diagnostics['rmse'],
        'q_loss': q_loss_value
    }

    return results


def compare_all_approaches(true_k=6, n_samples=250, n_features=18):
    """Compare all factor selection approaches using the same dataset"""

    print(f"\n{'='*80}")
    print("COMPARING ALL FACTOR SELECTION APPROACHES")
    print(f"{'='*80}")

    # Generate shared dataset
    dataset = get_shared_dataset(
        true_k=true_k,
        n_samples=n_samples,
        n_features=n_features,
        seed=123,
        force_regenerate=True
    )

    V = dataset['V']
    U = dataset['U']
    true_k = dataset['true_k']
    species_names = dataset['species_names']

    print(f"Using shared dataset for all approaches:")
    print(f"  - Samples: {V.shape[0]}")
    print(f"  - Species: {V.shape[1]}")
    print(f"  - True factors: {true_k}")
    print(f"  - Total mass range: [{V.sum(axis=1).min():.1f}, {V.sum(axis=1).max():.1f}]")

    # Compare approaches using the exact same dataset
    print(f"\nRunning comparison with consistent dataset...")
    results = compare_factor_selection_approaches(
        X=V,
        X_uncertainty=U,
        max_factors=min(12, true_k + 4),
        species_names=species_names
    )

    # Enhanced summary table
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS - SAME DATASET")
    print(f"{'='*80}")
    print(f"{'Approach':<20} {'Optimal K':<10} {'True K':<8} {'Accuracy':<10} {'R²':<8} {'Q-loss':<10}")
    print(f"{'-'*80}")

    comparison_summary = {}
    for approach, result in results.items():
        if 'error' in result:
            print(f"{approach:<20} {'ERROR':<10} {true_k:<8} {'FAILED':<10} {'N/A':<8} {'N/A':<10}")
            comparison_summary[approach] = {'status': 'failed', 'error': result['error']}
        else:
            optimal_k = result['optimal_factors']
            accuracy = "✓" if optimal_k == true_k else "✗"
            r2 = result['model_diagnostics'].get('r2', 0)

            # Calculate Q-loss for this approach
            # Note: This assumes the model results are available
            try:
                model = result.get('model')  # May not be available in compare function
                if hasattr(model, 'factor_contributions'):
                    q_loss_val = q_loss(V, U, model.factor_contributions, model.factor_profiles)
                else:
                    q_loss_val = 0.0
            except:
                q_loss_val = 0.0

            print(f"{approach:<20} {optimal_k:<10} {true_k:<8} {accuracy:<10} {r2:<8.2f} {q_loss_val:<10.1f}")

            comparison_summary[approach] = {
                'status': 'success',
                'optimal_factors': optimal_k,
                'accuracy': optimal_k == true_k,
                'r2': r2,
                'q_loss': q_loss_val
            }

    # Best performing approach
    successful_approaches = {k: v for k, v in comparison_summary.items() if v['status'] == 'success'}
    if successful_approaches:
        best_approach = max(successful_approaches.keys(),
                          key=lambda k: (successful_approaches[k]['accuracy'], successful_approaches[k]['r2']))
        print(f"\nBest performing approach: {best_approach.upper()}")
        best_result = successful_approaches[best_approach]
        print(f"  - Correct factor count: {best_result['accuracy']}")
        print(f"  - R²: {best_result['r2']:.3f}")
        print(f"  - Q-loss: {best_result['q_loss']:.1f}")

    return results, comparison_summary


def demonstrate_ard_details():
    """Detailed demonstration of ARD approach with weak factors"""

    print(f"\n{'='*80}")
    print("DETAILED ARD (AUTOMATIC RELEVANCE DETERMINATION) ANALYSIS")
    print(f"{'='*80}")

    # Generate data with specific factor characteristics using the real generator
    print("Generating dataset with varying factor strengths...")

    # First generate a base dataset
    V_base, U_base, true_k, true_H, true_W, simulator = generate_dataset(
        true_k=5, n_samples=400, n_features=25, seed=42
    )

    # Modify the dataset to have weak factors
    print("Modifying dataset to include weak factors...")

    # Get the true profiles and contributions from simulator
    true_H = simulator.syn_profiles  # Shape: (factors, species)
    true_W = simulator.syn_contributions  # Shape: (samples, factors)

    # Weaken some factors
    if true_k >= 4:
        true_W[:, -2] *= 0.3  # Second to last factor - weak
        true_H[-2, :] *= 0.3
    if true_k >= 5:
        true_W[:, -1] *= 0.1  # Last factor - very weak
        true_H[-1, :] *= 0.1

    # Regenerate V with modified factors
    V = true_W @ true_H
    # Add noise consistent with original uncertainty structure
    noise_level = 0.05
    V += np.random.RandomState(42).normal(0, noise_level * V, V.shape)
    V = np.maximum(V, 0.01)  # Ensure positivity
    U = U_base  # Use original uncertainty structure

    factor_strengths = np.sum(true_W, axis=0)
    print(f"Created dataset with {true_k} factors:")
    for i, strength in enumerate(factor_strengths):
        strength_label = "STRONG" if strength > factor_strengths.mean() else "WEAK"
        print(f"  Factor {i+1}: {strength:.1f} ({strength_label})")

    # Fit ARD model with more factors than true to test pruning
    model = HierarchicalBayesianNMF(
        max_factors=min(15, true_k + 8),  # More than true factors
        approach="ard",
        seed=42
    )

    species_names = [f'Species_{i+1}' for i in range(V.shape[1])]

    print(f"\nFitting ARD model with up to {model.max_factors} factors...")
    print(f"(True number of factors: {true_k})")

    model.fit(
        X=V,
        X_uncertainty=U,
        species_names=species_names,
        n_samples=2000,
        n_tune=1000,
        n_chains=4
    )

    print(f"\nARD Results:")
    print(f"True factors: {true_k}")
    print(f"ARD detected: {model.optimal_factors} active factors")

    # Show detailed factor relevance analysis using the new diagnostics
    if hasattr(model, 'factor_diagnostics'):
        diagnostics = model.factor_diagnostics
        print(f"\nEnhanced Factor Analysis (Weighted Voting System):")
        print(f"{'Factor':<8} {'Prec_W':<8} {'Prec_H':<8} {'Strength':<10} {'Contrib%':<9} {'Import%':<9} {'SNR':<8} {'Coher':<8} {'WtVotes':<9} {'Score':<8} {'Status':<8} {'True':<8}")
        print(f"{'-'*125}")

        for i in range(len(diagnostics['lambda_W_mean'])):
            prec_w = diagnostics['lambda_W_mean'][i]
            prec_h = diagnostics['lambda_H_mean'][i]
            strength = diagnostics['combined_strength'][i]
            contrib_pct = diagnostics['relative_contribution'][i] * 100
            import_pct = diagnostics['factor_importance'][i] * 100
            snr = diagnostics['factor_snr'][i]
            coherence = diagnostics['profile_coherence'][i]
            weighted_votes = diagnostics['weighted_votes'][i]
            score = diagnostics['factor_scores'][i]
            status = "ACTIVE" if model.selection_result.factor_probabilities[i] > 0.5 else "pruned"
            true_strength = factor_strengths[i] if i < len(factor_strengths) else 0.0

            print(f"{i+1:<8} {prec_w:<8.2f} {prec_h:<8.2f} {strength:<10.1f} {contrib_pct:<9.2f} {import_pct:<9.2f} {snr:<8.2f} {coherence:<8.2f} {weighted_votes:<9.1f} {score:<8.3f} {status:<8} {true_strength:<8.1f}")

        print(f"\nWeighted Voting System:")
        vote_weights = diagnostics['vote_weights']
        total_votes = sum(vote_weights.values())
        vote_threshold = diagnostics['thresholds']['vote_threshold']
        print(f"  - Vote weights: Precision={vote_weights['precision']}, Strength={vote_weights['strength']}, Contribution={vote_weights['contribution']}")
        print(f"  - Vote weights: Importance={vote_weights['importance']}, SNR={vote_weights['snr']}, Coherence={vote_weights['coherence']}")
        print(f"  - Total possible votes: {total_votes}")
        print(f"  - Threshold for selection: {vote_threshold:.1f} ({vote_threshold/total_votes:.1%} of total)")

        print(f"\nEnhanced Selection Thresholds:")
        thresholds = diagnostics['thresholds']
        print(f"  - Precision W/H (80th percentile): {thresholds['precision_W']:.2f} / {thresholds['precision_H']:.2f}")
        print(f"  - Strength threshold (60th percentile): {thresholds['strength']:.1f}")
        print(f"  - Mass threshold (60th percentile): {thresholds['mass']:.1f}")
        print(f"  - Contribution threshold: {thresholds['contribution']*100:.1f}%")
        print(f"  - Importance threshold: {thresholds['importance']*100:.1f}%")
        print(f"  - SNR threshold (50th percentile): {thresholds['snr']:.2f}")
        print(f"  - Coherence threshold (50th percentile): {thresholds['coherence']:.2f}")
        print(f"  - Max reasonable factors: {thresholds['max_reasonable']}")
        print(f"  - Min required factors: {thresholds['min_required']}")

        print(f"\nVoting Breakdown (Binary Votes):")
        print(f"  - Precision votes: {np.sum(diagnostics['precision_active'])} factors")
        print(f"  - Strength votes: {np.sum(diagnostics['strength_active'])} factors")
        print(f"  - Contribution votes: {np.sum(diagnostics['contribution_active'])} factors")
        print(f"  - Importance votes: {np.sum(diagnostics['importance_active'])} factors")
        print(f"  - SNR votes: {np.sum(diagnostics['snr_active'])} factors")
        print(f"  - Coherence votes: {np.sum(diagnostics['coherence_active'])} factors")

        # Enhanced performance analysis
        if len(model.selection_result.factor_probabilities) >= true_k:
            active_factors = model.selection_result.factor_probabilities > 0.5

            # True positive rate (correctly identified true factors)
            true_factors_kept = np.sum(active_factors[:true_k])
            true_positive_rate = true_factors_kept / true_k

            # False positive rate (incorrectly identified extra factors)
            if len(active_factors) > true_k:
                extra_factors_kept = np.sum(active_factors[true_k:])
                false_positive_rate = extra_factors_kept / (len(active_factors) - true_k)
            else:
                false_positive_rate = 0.0

            # Factor strength correlation
            if true_k <= len(active_factors):
                true_strength_subset = factor_strengths[:min(true_k, len(factor_strengths))]
                detected_strength_subset = diagnostics['combined_strength'][:len(true_strength_subset)]
                if len(true_strength_subset) > 1 and len(detected_strength_subset) > 1:
                    from scipy.stats import pearsonr
                    strength_corr, _ = pearsonr(true_strength_subset, detected_strength_subset)
                else:
                    strength_corr = 0.0
            else:
                strength_corr = 0.0

            # Factor importance analysis
            importance_analysis = []
            for i in range(min(len(active_factors), len(factor_strengths))):
                true_str = factor_strengths[i] if i < len(factor_strengths) else 0.0
                detected_imp = diagnostics['factor_importance'][i] * 100
                detected_contrib = diagnostics['relative_contribution'][i] * 100
                is_active = active_factors[i]
                should_be_active = true_str > 0

                importance_analysis.append({
                    'factor': i + 1,
                    'true_strength': true_str,
                    'detected_importance': detected_imp,
                    'detected_contribution': detected_contrib,
                    'is_active': is_active,
                    'should_be_active': should_be_active,
                    'correct': is_active == should_be_active
                })

            correct_classifications = sum(1 for item in importance_analysis if item['correct'])
            total_classifications = len(importance_analysis)
            classification_accuracy = correct_classifications / total_classifications if total_classifications > 0 else 0

            print(f"\nEnhanced ARD Performance Metrics:")
            print(f"  - True factors correctly identified: {true_factors_kept}/{true_k} ({true_positive_rate:.2%})")
            print(f"  - Extra factors incorrectly kept: {extra_factors_kept if len(active_factors) > true_k else 0}/{len(active_factors) - true_k if len(active_factors) > true_k else 0}")
            print(f"  - False positive rate: {false_positive_rate:.2%}")
            print(f"  - Factor strength correlation: {strength_corr:.3f}")
            print(f"  - Classification accuracy: {classification_accuracy:.2%} ({correct_classifications}/{total_classifications})")
            print(f"  - Precision (true positives): {true_positive_rate:.2%}")
            print(f"  - Selection efficiency: {1 - false_positive_rate:.2%}")

            # Show factor-by-factor analysis
            print(f"\nFactor-by-Factor Analysis:")
            print(f"{'Factor':<8} {'True Str':<10} {'Det Import%':<12} {'Det Contrib%':<13} {'Active':<8} {'Should Be':<10} {'Correct':<8}")
            print(f"{'-'*75}")
            for item in importance_analysis:
                print(f"{item['factor']:<8} {item['true_strength']:<10.1f} {item['detected_importance']:<12.2f} {item['detected_contribution']:<13.2f} {str(item['is_active']):<8} {str(item['should_be_active']):<10} {str(item['correct']):<8}")
    else:
        # Fallback to original relevance display
        relevance = model.selection_result.factor_probabilities
        print(f"\nFactor Relevance Analysis:")
        print(f"{'Factor':<8} {'Relevance':<12} {'Status':<10} {'True Strength':<15}")
        print(f"{'-'*50}")

        for i, rel in enumerate(relevance):
            status = "ACTIVE" if rel > 0.5 else "pruned"
            true_strength = factor_strengths[i] if i < len(factor_strengths) else 0.0
            print(f"{i+1:<8} {rel:.3f}{'':7} {status:<10} {true_strength:.1f}")

    # Calculate final metrics
    q_loss_value = q_loss(V, U, model.factor_contributions, model.factor_profiles)
    print(f"\nFinal Model Performance:")
    print(f"  - R²: {model.model_diagnostics['r2']:.3f}")
    print(f"  - Q-loss: {q_loss_value:.1f}")
    accuracy_symbol = "✓" if model.optimal_factors == true_k else "✗"
    print(f"  - Factor selection accuracy: {accuracy_symbol}")

    try:
        model.plot_factor_selection_results()
    except Exception as e:
        print(f"Plotting failed: {e}")

    return model


if __name__ == "__main__":
    print("Hierarchical Bayesian NMF Factor Selection Demonstration")
    print("Using consistent datasets from esat-net utils.generate_dataset()")
    print("=" * 60)

    # 1. Demonstrate ARD approach in detail
    print("\n" + "="*60)
    print("PHASE 1: DETAILED ARD ANALYSIS")
    print("="*60)
    model_ard = demonstrate_ard_details()

    # 2. Demonstrate discrete search approach using shared dataset
    print("\n" + "="*60)
    print("PHASE 2: DISCRETE SEARCH APPROACH")
    print("="*60)
    model_discrete = demonstrate_single_approach("discrete_search", use_shared=True, true_k=4)

    # 3. Compare all approaches on the same dataset
    print("\n" + "="*60)
    print("PHASE 3: COMPREHENSIVE COMPARISON")
    print("="*60)
    comparison_results, summary = compare_all_approaches(true_k=5, n_samples=300, n_features=20)

    print(f"\n{'='*80}")
    print("DEMONSTRATION COMPLETE - SUMMARY")
    print(f"{'='*80}")

    print("\nKey Findings:")
    successful_methods = [k for k, v in summary.items() if v['status'] == 'success']
    accurate_methods = [k for k, v in summary.items() if v.get('accuracy', False)]

    print(f"1. Successfully completed methods: {len(successful_methods)}/{len(summary)}")
    print(f"   Methods: {', '.join(successful_methods)}")

    if accurate_methods:
        print(f"2. Methods with correct factor count: {len(accurate_methods)}")
        print(f"   Methods: {', '.join(accurate_methods)}")

        # Find best performing accurate method
        best_accurate = max(accurate_methods,
                          key=lambda k: summary[k]['r2'])
        print(f"3. Best accurate method: {best_accurate.upper()}")
        print(f"   R²: {summary[best_accurate]['r2']:.3f}")
    else:
        print("2. No methods achieved correct factor count")

        # Find best performing method overall
        if successful_methods:
            best_overall = max(successful_methods,
                             key=lambda k: summary[k]['r2'])
            print(f"3. Best performing method: {best_overall.upper()}")
            print(f"   Factors: {summary[best_overall]['optimal_factors']} (true: varies)")
            print(f"   R²: {summary[best_overall]['r2']:.3f}")

    print(f"\nApproach Recommendations:")
    print("- ARD: Best for automatic factor pruning with unknown factor counts")
    print("- Discrete Search: Best for systematic comparison across factor ranges")
    print("- Nested Comparison: Best for robust validation with cross-validation")
    print("- Infinite: Best for completely unknown scenarios with flexible priors")
    print(f"\nAll approaches used the same esat-net synthetic datasets for fair comparison.")
