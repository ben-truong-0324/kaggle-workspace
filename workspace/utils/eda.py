import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.covariance import EllipticEnvelope

from scipy.stats import skew, kurtosis, mannwhitneyu, ks_2samp, chi2_contingency, ttest_ind, levene, shapiro
from statsmodels.stats.proportion import proportion_confint

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

def eda_vis(X, y, task_type):
    print("📊 Target Column Summary:")
    print(f"  Shape: {y.shape}")
    print(f"  Unique values: {y.unique().tolist() if y.nunique() < 30 else 'Too many to display'}")
    print(f"  Value counts:")
    display(y.value_counts().sort_index())

    sns_palette = "Set2"
    
    # === Basic Target Distribution & Relative Frequency (side-by-side) ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    if task_type in ["binary_classification", "multiclass_classification"]:
        sns.countplot(ax=axes[0], x=y, hue=y,palette=sns_palette, order=sorted(y.unique()))
        axes[0].set_title("Target Class Distribution")
        axes[0].set_xlabel("Target Class")
        axes[0].set_ylabel("Count")
    
        value_counts = y.value_counts(normalize=True).sort_index()
        sns.barplot(ax=axes[1], x=value_counts.index,palette=sns_palette, y=value_counts.values)
        axes[1].set_title("Relative Frequency of Target Classes")
        axes[1].set_ylabel("Proportion")
        axes[1].set_xlabel("Target Class")
        axes[1].set_ylim(0, 1)
    
    elif task_type == "regression":
        sns.histplot(ax=axes[0], data=y, kde=True)
        axes[0].set_title("Target Distribution (Histogram & KDE)")
        axes[0].set_xlabel("Target")
        axes[0].set_ylabel("Density")
        axes[1].axis('off')  # No second plot for regression here
    
    plt.tight_layout()
    plt.show()
    
    # === Top 12 Correlated Features with y (Scatter Grid) ===
    if task_type == "regression" or pd.api.types.is_numeric_dtype(y):
        df_corr = pd.concat([X.select_dtypes(include=np.number), y.rename("target")], axis=1)
        corr_vals = df_corr.corr()["target"].drop("target").abs().sort_values(ascending=False)
        top12 = corr_vals.head(12).index
    
        n_cols = 4
        n_rows = 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
        for i, col in enumerate(top12):
            ax = axes[i // n_cols, i % n_cols]
            sns.scatterplot(ax=ax, x=X[col], y=y, hue=y, palette=sns_palette, legend=False)
            ax.set_title(f"{col} vs Target (r={corr_vals[col]:.2f})")
            ax.set_xlabel(col)
            ax.set_ylabel("Target")
    
        # Remove any unused subplots if top12 has < 12
        for j in range(i + 1, n_cols * n_rows):
            fig.delaxes(axes[j // n_cols, j % n_cols])
    
        plt.tight_layout()
        plt.show()
    
    # === Boxplots ===
    print("Boxplots by Target (only showing 12 features):")
    
    num_cols = X.select_dtypes(include=np.number).columns[:12]  # Limit to 12
    n_cols = 3
    n_rows = int(np.ceil(len(num_cols) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
    for i, col in enumerate(num_cols):
        ax = axes[i // n_cols, i % n_cols]
        sns.boxplot(x=y, y=X[col], ax=ax, hue=y,palette=sns_palette)
        ax.set_title(f"{col} by Target")
        ax.set_xlabel("Target")
        ax.set_ylabel(col)
    
    # Hide unused axes
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes[j // n_cols, j % n_cols])
    
    plt.tight_layout()
    plt.show()


def eda_feature_transformation_check(X, y, task_type, contamination=0.01):
    """
    Suggest transformations for each feature based on distribution, type, and target relationship.
    Returns structured dict for use in ETL pipelines (e.g., DAGshub).
    """
    recommendations = {}
    X_num = X.select_dtypes(include=np.number).copy()
    X_cat = X.select_dtypes(include=['object', 'category']).copy()

    # Compute mutual information
    if task_type == "regression":
        mi = mutual_info_regression(X_num.fillna(0), y)
    else:
        mi = mutual_info_classif(X_num.fillna(0), y)
    mi_scores = pd.Series(mi, index=X_num.columns)

    # Elliptic Envelope Outlier Detection
    outlier_flags = None
    if len(X_num.columns) > 1:
        try:
            ee = EllipticEnvelope(contamination=contamination, random_state=42)
            ee.fit(X_num.fillna(0))
            outlier_flags = ee.predict(X_num.fillna(0)) == -1
        except Exception as e:
            print(f"⚠️ EllipticEnvelope failed: {e}")
            outlier_flags = None

    for col in X.columns:
        feature = X[col]
        feature_type = feature.dtype
        nunique = feature.nunique(dropna=True)
        std = feature.std()
        mean = feature.mean() if np.issubdtype(feature_type, np.number) else None
        range_ = feature.max() - feature.min() if np.issubdtype(feature_type, np.number) else None
        feature_skew = skew(feature.dropna()) if np.issubdtype(feature_type, np.number) else None

        suggestions = {}

        # Drop constant
        if nunique <= 1 or (np.issubdtype(feature_type, np.number) and np.isclose(std, 0.0)):
            suggestions["suggestion_1"] = "drop"
        elif feature_type == "object" or feature_type.name == "category":
            if nunique <= 10:
                suggestions["suggestion_1"] = "one_hot_encode"
            else:
                suggestions["suggestion_1"] = "label_encode"
        elif np.issubdtype(feature_type, np.number):
            corr = np.corrcoef(feature.fillna(0), y)[0, 1] if len(y) == len(feature) else 0
            mi_score = mi_scores.get(col, 0)

            # Primary transformation
            if abs(feature_skew) > 1 and abs(corr) > 0.2:
                suggestions["suggestion_1"] = "log_transform"
            elif range_ and range_ > 100:
                suggestions["suggestion_1"] = "minmax_scale"
            elif abs(corr) < 0.05 and mi_score < 0.01:
                suggestions["suggestion_1"] = "drop"
            else:
                suggestions["suggestion_1"] = "standard_scale"

            # Secondary: Outlier sensitivity
            if outlier_flags is not None and outlier_flags.sum() > 0:
                outlier_contrib = feature[outlier_flags]
                if outlier_contrib.std() > 2 * std:
                    suggestions["suggestion_2"] = "check_outliers"

        else:
            suggestions["suggestion_1"] = "inspect_manually"

        recommendations[col] = suggestions

    return recommendations

   
# === Exploratory Data Analysis X features ===
def feature_eda_vis(X, y, task_type):
    print("📊 Feature Statistics:")
    # === Feature Diagnostics ===
    for col in X.columns:
        try:
            feature = X[col]
            if feature.dtype.kind in "bifc" and feature.notnull().sum() > 0:  # skip text or empty
                if feature.dtype == bool:
                    feature = feature.astype(int)
                
                # Create side-by-side plots
                fig, axes = plt.subplots(1, 3, figsize=(18, 4))

                # Plot 1: Histogram
                sns.histplot(feature, kde=True, bins=20, ax=axes[0])
                axes[0].set_title(f"Histogram: {col}")
                axes[0].set_xlabel(col)

                # Plot 2: y vs. feature (boxplot or scatter depending on feature type)
                x_binned = pd.cut(feature, bins=10) if feature.nunique() > 10 else feature

                # For classification: show frequency of target class per bin
                if "classification" in task_type:
                    heatmap_data = pd.crosstab(x_binned, y)
                    heatmap_data.columns = [str(col) for col in heatmap_data.columns]  # Clean column names
                    
                else:
                    y_binned_raw = pd.qcut(y, q=5, duplicates="drop")
                    if y_binned_raw.nunique() <= 1:
                        y_binned_raw = pd.cut(y, bins=5)
                    y_binned = y_binned_raw.astype("category").cat.codes
                    heatmap_data = pd.crosstab(x_binned, y_binned)
                    heatmap_data.columns = [str(col) for col in heatmap_data.columns]


                sns.heatmap(heatmap_data, cmap="viridis", annot=True, fmt="d", ax=axes[1])
                axes[1].set_title(f"Heatmap: {col} vs Target")
                axes[1].set_xlabel("Target" if task_type == "classification" else "Binned Target")
                axes[1].set_ylabel(f"{col} Bins")

                # Plot 3: Deviation from 50% for classification
                if "classification" in task_type:
                    target_true = y # == 1  # assumes binary target
                    df_temp = pd.DataFrame({col: x_binned, "target": target_true})
                    bin_stats = df_temp.groupby(col)["target"].agg(["count", "sum"])
                    bin_stats["rate"] = bin_stats["sum"] / bin_stats["count"]
                    bin_stats["deviation"] = (bin_stats["rate"] ) * 100 - 100

                    deviations = bin_stats["deviation"]
                    # colors = plt.cm.seismic((deviations - deviations.min()) / (deviations.max() - deviations.min()))
                    deviation_vals = deviations.to_numpy(dtype=float)  # ensure numeric
                    normed = (deviation_vals - deviation_vals.min()) / (deviation_vals.max() - deviation_vals.min() + 1e-8)
                    colors = plt.cm.seismic(normed)
                    axes[2].bar(bin_stats.index.astype(str), deviations, color=colors)

                    # bin_stats["deviation"].plot(kind="bar", ax=axes[2])

                    axes[2].set_title(f"Deviation from 50% True Rate\n({col})")
                    axes[2].set_ylabel("Deviation (%)")
                    axes[2].axhline(0, color="gray", linestyle="--")
                    axes[2].tick_params(axis='x', rotation=45)
                else:
                    axes[2].set_visible(False)

                plt.tight_layout()
                plt.show()

                mean = feature.mean()
                std = feature.std()
                skewness = skew(feature.dropna())
                kurt = kurtosis(feature.dropna())
                min_val, max_val = feature.min(), feature.max()
                zero_variance = np.isclose(std, 0.0)

                # Correlation with target
                try:
                    target_corr = np.corrcoef(feature, y)[0, 1]
                except Exception:
                    target_corr = float("nan")

                # Max correlation with other features
                corr_with_others = X.corr()[col].drop(col)
                most_corr_feat = corr_with_others.abs().idxmax()
                most_corr_val = corr_with_others[most_corr_feat]

                # === Preprocessing Suggestions ===
                suggestions = []
                if zero_variance:
                    suggestions.append("🔻 Drop (zero variance)")
                elif std > 50 or abs(mean) > 100 or max_val - min_val > 100:
                    suggestions.append("🔄 Normalize (high range)")
                if abs(skewness) > 1 or abs(kurt) > 3:
                    suggestions.append("🌀 Transform (non-Gaussian)")
                if feature.dtype == "object" or feature.nunique() < 10:
                    suggestions.append("📦 Encode (categorical or discrete)")
                if abs(target_corr) < 0.05:
                    suggestions.append("🤔 Low correlation with target")

                print(f"📈 {col} Stats:")
                print(f"  Mean: {mean:.3f}, Std: {std:.3f}, Skew: {skewness:.3f}, Kurtosis: {kurt:.3f}")
                print(f"  Correlation with target: {target_corr:.3f}")
                print(f"  Most correlated with: {most_corr_feat} ({most_corr_val:.3f})")
                print(f"🔧 Suggestions: {' | '.join(suggestions) if suggestions else '✅ None'}\n")
        except Exception as e:
            print(e)
            pass

    # === Feature Correlation Heatmap ===
    plt.figure(figsize=(12, 8))
    sns.heatmap(X.corr(), cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # === Correlation with Target ===
    Xy_corr = X.copy()
    Xy_corr["target"] = y
    target_corr = Xy_corr.corr()["target"].drop("target").sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=target_corr.values, y=target_corr.index)
    plt.title("Correlation of Features with Target")
    plt.xlabel("Correlation")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # === Projections (PCA, ICA, t-SNE) ===
    X_scaled = X.select_dtypes(include=np.number).fillna(0)

    # ICA
    ica = FastICA(n_components=2, random_state=42)
    ica_proj = ica.fit_transform(X_scaled)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=ica_proj[:, 0], y=ica_proj[:, 1], hue=y, palette='tab10', s=40, legend=False)
    plt.title("ICA Projection")
    plt.xlabel("IC1")
    plt.ylabel("IC2")
    plt.tight_layout()
    plt.show()

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_proj = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=tsne_proj[:, 0], y=tsne_proj[:, 1], hue=y, palette='tab10', s=40, legend=False)
    plt.title("t-SNE Projection")
    plt.tight_layout()
    plt.show()

    # PCA
    pca_vis = PCA(n_components=2, random_state=42)
    X_vis = pca_vis.fit_transform(X_scaled)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1], hue=y, palette='tab10', s=30, legend=False)
    plt.title("PCA Projection of Features (colored by y)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

    # === Mutual Information with Target ===
    if task_type == "regression":
        mi = mutual_info_regression(X_scaled, y)
    else:
        mi = mutual_info_classif(X_scaled, y)

    mi_series = pd.Series(mi, index=X_scaled.columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=mi_series.values, y=mi_series.index)
    plt.title("Mutual Information Between Features and Target")
    plt.xlabel("Mutual Information")
    plt.tight_layout()
    plt.show()


def eda_vis_v2(X,y):
    """
    #  X and y are pandas DataFrames/Series 
    """
    
    # --- Data Preparation ---
    y_named = y.copy()
    if not hasattr(y_named, 'name') or y_named.name is None:
        y_named.name = 'Transported' # Default name if y has no name

    df_combined = X.copy()
    df_combined['Transported_numeric'] = y_named.astype(int)

    CATEGORICAL_THRESHOLD = 20 
    palette = {0: 'skyblue', 1: 'salmon'} 
    legend_labels = {0: 'Not Transported', 1: 'Transported'}
    feature_columns = X.columns

    # --- Main Loop for Plotting ---
    for feature_col in feature_columns:
        print(f"--- Analyzing Feature: {feature_col} ---")
        
        fig, axes = plt.subplots(2, 3, figsize=(22, 13)) # Adjusted figsize slightly for better label spacing
        fig.suptitle(f'Comprehensive Analysis: {feature_col} | Target: {y_named.name}', fontsize=18, y=0.99) # Adjusted y for suptitle
        
        # --- Row 1: Distribution Visualizations ---
        ax_1_1 = axes[0, 0]
        try:
            plot_data_1_1 = df_combined[[feature_col, 'Transported_numeric']].dropna(subset=[feature_col])
            if not plot_data_1_1.empty and plot_data_1_1[feature_col].nunique() > 0 :
                sns.stripplot(data=plot_data_1_1, x=feature_col, y='Transported_numeric', hue='Transported_numeric',
                            jitter=0.25, dodge=True, ax=ax_1_1, palette=palette, legend=False, alpha=0.6)
                ax_1_1.set_title('Point Distribution by Target')
                ax_1_1.set_ylabel(f'{y_named.name} (0=F, 1=T)')
                ax_1_1.set_yticks([0, 1])
                ax_1_1.set_yticklabels(['False', 'True'])
                ax_1_1.set_xlabel(feature_col)
                handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=10) for i in palette]
                ax_1_1.legend(handles, [legend_labels[i] for i in palette], title=y_named.name, loc='best')
                
                # Rotate x-axis labels if needed
                if plot_data_1_1[feature_col].nunique() > 5 and plot_data_1_1[feature_col].dtype == 'object': # More categories
                    ax_1_1.tick_params(axis='x', rotation=45, labelbottom=True)
                    plt.setp(ax_1_1.get_xticklabels(), ha="right", rotation_mode="anchor")
                elif plot_data_1_1[feature_col].nunique() > 10 and plot_data_1_1[feature_col].dtype != 'object': # Numerical with many ticks
                    ax_1_1.tick_params(axis='x', rotation=30, labelbottom=True)
                    plt.setp(ax_1_1.get_xticklabels(), ha="right", rotation_mode="anchor")
                else: # Fewer categories or numerical, no rotation or default handling
                    ax_1_1.tick_params(axis='x', labelbottom=True)

            else:
                ax_1_1.text(0.5, 0.5, "No data or no variance\nafter NaN drop", ha='center', va='center', transform=ax_1_1.transAxes)
                ax_1_1.set_title('Point Distribution (No Data)')
        except Exception as e:
            ax_1_1.set_title('Point Distribution (Error)')
            ax_1_1.text(0.5, 0.5, f"Plot failed: {e}", ha='center', va='center', transform=ax_1_1.transAxes, wrap=True)
            print(f"  Error in Plot 1.1 for {feature_col}: {e}")

        ax_1_2 = axes[0, 1]
        try:
            plot_data_1_2 = df_combined[[feature_col, 'Transported_numeric']].dropna(subset=[feature_col])
            if not plot_data_1_2.empty and plot_data_1_2[feature_col].nunique() > 0:
                sns.histplot(data=plot_data_1_2, x=feature_col, hue='Transported_numeric', 
                            multiple='layer', kde=False, ax=ax_1_2, palette=palette, 
                            stat="density", common_norm=False, alpha=0.6, legend=True)
                ax_1_2.set_title('Normalized Histogram by Target')
                ax_1_2.set_xlabel(feature_col)
                handles_hist, labels_hist = ax_1_2.get_legend_handles_labels()
                try: 
                    labels_hist_descriptive = [legend_labels[int(float(l))] for l in labels_hist]
                    ax_1_2.legend(handles_hist, labels_hist_descriptive, title=y_named.name)
                except (ValueError, KeyError): 
                    ax_1_2.legend(title=y_named.name)
            else:
                ax_1_2.text(0.5, 0.5, "No data or no variance\nafter NaN drop", ha='center', va='center', transform=ax_1_2.transAxes)
                ax_1_2.set_title('Histogram (No Data)')
        except Exception as e:
            ax_1_2.set_title('Histogram (Error)')
            ax_1_2.text(0.5, 0.5, f"Plot failed: {e}", ha='center', va='center', transform=ax_1_2.transAxes, wrap=True)
            print(f"  Error in Plot 1.2 for {feature_col}: {e}")

        ax_1_3 = axes[0, 2]
        try:
            temp_df_combined = df_combined.dropna(subset=[feature_col]) 
            if not temp_df_combined.empty and temp_df_combined[feature_col].nunique() > 0:
                is_categorical_like = temp_df_combined[feature_col].dtype == 'object' or \
                                    temp_df_combined[feature_col].nunique() < CATEGORICAL_THRESHOLD
                
                target_0_data = temp_df_combined[temp_df_combined['Transported_numeric'] == 0][feature_col]
                target_1_data = temp_df_combined[temp_df_combined['Transported_numeric'] == 1][feature_col]

                if not target_0_data.empty and not target_1_data.empty:
                    if is_categorical_like:
                        props_0 = target_0_data.value_counts(normalize=True)
                        props_1 = target_1_data.value_counts(normalize=True)
                        all_categories = sorted(list(set(props_0.index) | set(props_1.index)))
                        props_0 = props_0.reindex(all_categories, fill_value=0)
                        props_1 = props_1.reindex(all_categories, fill_value=0)
                        diff_props = props_1 - props_0
                        diff_props.plot(kind='bar', ax=ax_1_3, color=['tomato' if x < 0 else 'mediumseagreen' for x in diff_props.values]) # Removed rot here
                        ax_1_3.tick_params(axis='x', rotation=45) # Apply rotation
                        plt.setp(ax_1_3.get_xticklabels(), ha='right', rotation_mode='anchor') # Ensure alignment
                        ax_1_3.set_ylabel('Prop(Y=1) - Prop(Y=0)')
                    else: 
                        min_val = min(target_0_data.min(), target_1_data.min())
                        max_val = max(target_0_data.max(), target_1_data.max())
                        if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val: # Handle NaN or single value case
                            bins = np.array([temp_df_combined[feature_col].min() - 0.5, temp_df_combined[feature_col].max() + 0.5]) if temp_df_combined[feature_col].nunique() > 0 else np.array([0,1])
                        else: 
                            bins = np.linspace(min_val, max_val, 11)
                        hist_0, _ = np.histogram(target_0_data.dropna(), bins=bins, density=True) # dropna within histogram
                        hist_1, _ = np.histogram(target_1_data.dropna(), bins=bins, density=True) # dropna within histogram
                        diff_hist = hist_1 - hist_0
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        bar_width = bins[1] - bins[0] if len(bins) > 1 else 1
                        ax_1_3.bar(bin_centers, diff_hist, width=bar_width * 0.9, 
                                    color=['tomato' if x < 0 else 'mediumseagreen' for x in diff_hist])
                        ax_1_3.set_ylabel('Density(Y=1) - Density(Y=0)')
                    ax_1_3.axhline(0, color='black', lw=0.8, linestyle='--')
                    ax_1_3.set_title('Outcome Difference by Feature Value')
                    ax_1_3.set_xlabel(feature_col)
                else:
                    ax_1_3.text(0.5, 0.5, "Not enough data in one/both target groups", ha='center', va='center', transform=ax_1_3.transAxes)
                    ax_1_3.set_title('Outcome Difference (Not enough data)')
            else:
                ax_1_3.text(0.5, 0.5, "No data or no variance\nafter NaN drop", ha='center', va='center', transform=ax_1_3.transAxes)
                ax_1_3.set_title('Outcome Difference (No Data)')
        except Exception as e:
            ax_1_3.set_title('Outcome Difference (Error)')
            ax_1_3.text(0.5, 0.5, f"Plot failed: {e}", ha='center', va='center', transform=ax_1_3.transAxes, wrap=True)
            print(f"  Error in Plot 1.3 for {feature_col}: {e}")

        # --- Row 2: Statistical Significance ---
        ax_2_1 = axes[1, 0] 
        ax_2_2 = axes[1, 1] 
        ax_2_3 = axes[1, 2] 

        for ax_text in [ax_2_2, ax_2_3]:
            ax_text.clear()
            ax_text.axis('off')

        current_feature_data = df_combined[[feature_col, 'Transported_numeric']].dropna(subset=[feature_col])
        if current_feature_data.empty or current_feature_data[feature_col].nunique() == 0: # Also check for no variance
            ax_2_1.text(0.5, 0.5, "No data or no variance\nfor stats after NaN drop", ha='center', va='center', transform=ax_2_1.transAxes)
            ax_2_1.set_title('Stats (No Data)')
            ax_2_2.text(0.05, 0.9, "No data or no variance for stats after NaN drop.", fontsize=9, va='top', wrap=True)
            ax_2_3.text(0.05, 0.9, "No data or no variance for stats after NaN drop.", fontsize=9, va='top', wrap=True)
            fig.subplots_adjust(hspace=0.6, wspace=0.35, top=0.93, bottom=0.08, left=0.05, right=0.97)
            plt.show()
            continue 

        is_categorical_feature = current_feature_data[feature_col].dtype == 'object' or \
                                current_feature_data[feature_col].nunique() < CATEGORICAL_THRESHOLD
        
        stats_summary_text = ""
        interpretation_text = ""

        if is_categorical_feature:
            ax_2_1.set_title(f'Proportion Transported by {feature_col}\n(with 95% CIs)')
            # Ensure feature_col is treated as string for crosstab if it's not already object (e.g. boolean, int categories)
            contingency_table = pd.crosstab(current_feature_data[feature_col].astype(str), current_feature_data['Transported_numeric'])
            
            if contingency_table.shape[0] < 1 or contingency_table.shape[1] < 2 : # Need at least 1 category and 2 outcome classes
                stats_summary_text += "Chi-squared test not applicable (table too small or one outcome class missing).\n"
                ax_2_1.text(0.5,0.5, "Too few categories or\noutcomes for plot/test", ha='center', va='center', transform=ax_2_1.transAxes)
            elif contingency_table.shape[0] < 2: # Need at least 2 categories for chi2
                stats_summary_text += "Chi-squared test not applicable (needs at least 2 categories).\n"
                ax_2_1.text(0.5,0.5, "Needs at least 2 categories for Chi2 test", ha='center', va='center', transform=ax_2_1.transAxes)
            else:
                chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
                stats_summary_text += f"Chi-squared Test of Independence:\n"
                stats_summary_text += f"  Chi2 Stat: {chi2:.2f}, P-value: {p_chi2:.3g}\n  DOF: {dof}\n"
                interpretation_text += f"P-value ({p_chi2:.3g}) for Chi-squared test: "
                interpretation_text += "Suggests " + ("a significant" if p_chi2 < 0.05 else "no significant") + \
                                    f" association between {feature_col} and {y_named.name}.\n\n"

                categories = contingency_table.index
                proportions_transported = []
                ci_lows = []
                ci_highs = []
                
                for cat in categories:
                    count_transported = contingency_table.loc[cat, 1] if 1 in contingency_table.columns else 0
                    count_not_transported = contingency_table.loc[cat, 0] if 0 in contingency_table.columns else 0
                    n_obs_cat = count_transported + count_not_transported
                    if n_obs_cat > 0:
                        prop = count_transported / n_obs_cat
                        low, high = proportion_confint(count_transported, n_obs_cat, method='wilson')
                        proportions_transported.append(prop)
                        ci_lows.append(low)
                        ci_highs.append(high)
                    else: 
                        proportions_transported.append(0)
                        ci_lows.append(0)
                        ci_highs.append(0)

                prop_df = pd.DataFrame({
                    'category': categories, # Already strings due to .astype(str) in crosstab
                    'proportion_transported': proportions_transported,
                    'ci_low': ci_lows,
                    'ci_high': ci_highs
                })
                
                ax_2_1.bar(prop_df['category'], prop_df['proportion_transported'], 
                        yerr=[prop_df['proportion_transported'] - prop_df['ci_low'], prop_df['ci_high'] - prop_df['proportion_transported']],
                        capsize=5, color='mediumseagreen', alpha=0.7)
                ax_2_1.set_ylabel(f'Proportion {y_named.name}')
                ax_2_1.tick_params(axis='x', rotation=45) # Corrected: Apply rotation
                plt.setp(ax_2_1.get_xticklabels(), ha='right', rotation_mode='anchor') # Corrected: Ensure alignment
                ax_2_1.axhline(current_feature_data['Transported_numeric'].mean(), color='grey', linestyle='--', label='Overall Mean')
                if not prop_df.empty: # Only add legend if there's data to plot
                    ax_2_1.legend(loc='best')

                interpretation_text += "Error bars on plot show 95% CIs for proportion transported. "
                interpretation_text += "If CIs for different categories don't overlap much, "
                interpretation_text += "it suggests a significant difference in transport rates.\n"

        else: # Numerical feature
            group0 = current_feature_data[current_feature_data['Transported_numeric'] == 0][feature_col].dropna() # Ensure NaNs are out for tests
            group1 = current_feature_data[current_feature_data['Transported_numeric'] == 1][feature_col].dropna() # Ensure NaNs are out for tests

            ax_2_1.set_title(f'{feature_col} Distribution by Target')
            # For boxplot, ensure data passed has NaNs handled if seaborn version is older
            sns.boxplot(x='Transported_numeric', y=feature_col, data=current_feature_data.dropna(subset=[feature_col]), 
                        ax=ax_2_1, palette=palette, hue='Transported_numeric', legend=False)
            ax_2_1.set_xticklabels([legend_labels[0], legend_labels[1]])
            ax_2_1.set_xlabel(y_named.name)

            stats_summary_text += "Normality (Shapiro-Wilk):\n"
            norm_p0, norm_p1 = -1.0, -1.0 # Initialize as float
            if len(group0) >=3 : 
                shapiro_stat0, norm_p0 = shapiro(group0)
                stats_summary_text += f"  Group 0 (Not Transported): p={norm_p0:.3g}\n"
            else: stats_summary_text += "  Group 0: Too few samples for normality test.\n"
            if len(group1) >=3 :
                shapiro_stat1, norm_p1 = shapiro(group1)
                stats_summary_text += f"  Group 1 (Transported): p={norm_p1:.3g}\n"
            else: stats_summary_text += "  Group 1: Too few samples for normality test.\n"
            
            # Default to Mann-Whitney U if any group has < 3 samples for normality, or if normality fails
            use_ttest = (norm_p0 > 0.05 or len(group0) < 3) and \
                        (norm_p1 > 0.05 or len(group1) < 3)

            if use_ttest and len(group0)>1 and len(group1)>1:
                levene_stat, levene_p = levene(group0, group1)
                stats_summary_text += f"Homogeneity of Variances (Levene's test): p={levene_p:.3g}\n"
                equal_var = levene_p > 0.05

                t_stat, p_ttest = ttest_ind(group0, group1, equal_var=equal_var) # nan_policy='omit' is default in newer scipy
                stats_summary_text += f"Independent T-test (equal_var={equal_var}):\n"
                stats_summary_text += f"  T-statistic: {t_stat:.2f}, P-value: {p_ttest:.3g}\n"
                interpretation_text += f"P-value ({p_ttest:.3g}) from t-test: "
                interpretation_text += "Suggests " + ("a significant" if p_ttest < 0.05 else "no significant") + \
                                    f" difference in mean {feature_col} between groups.\n"
                ax_2_1.text(0.5, 0.95, f"T-test p-value: {p_ttest:.3g}", ha='center', va='top', transform=ax_2_1.transAxes, fontsize=9, color='red', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            elif len(group0)>0 and len(group1)>0: 
                try:
                    u_stat, p_mannwhitney = mannwhitneyu(group0, group1, alternative='two-sided') # nan_policy='omit' is default
                    stats_summary_text += f"Mann-Whitney U Test:\n"
                    stats_summary_text += f"  U-statistic: {u_stat:.0f}, P-value: {p_mannwhitney:.3g}\n"
                    interpretation_text += f"P-value ({p_mannwhitney:.3g}) from Mann-Whitney U test: "
                    interpretation_text += "Suggests " + ("a significant" if p_mannwhitney < 0.05 else "no significant") + \
                                    f" difference in distributions of {feature_col} between groups.\n"
                    ax_2_1.text(0.5, 0.95, f"Mann-Whitney p: {p_mannwhitney:.3g}", ha='center', va='top', transform=ax_2_1.transAxes, fontsize=9, color='red', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                except ValueError as e_mw: 
                    stats_summary_text += f"Mann-Whitney U Test: Error - {e_mw}\n"
                    interpretation_text += "Mann-Whitney U test could not be performed (e.g., identical data in groups).\n"
            else:
                stats_summary_text += "Not enough data in one or both groups for numerical tests.\n"
                interpretation_text += "Not enough data to compare groups statistically.\n"

        interpretation_text += "\nBayesian Perspective:\nA Bayesian approach could provide posterior distributions for parameters "
        interpretation_text += "(e.g., difference in means/proportions). This offers a richer view of uncertainty "
        interpretation_text += "and allows direct probability statements about the effect size, rather than just a p-value."

        ax_2_2.text(0.01, 0.98, stats_summary_text, fontsize=8, va='top', ha='left', wrap=True, family='monospace') # Reduced font size
        ax_2_2.set_title("Statistical Test Details", fontsize=10)
        
        ax_2_3.text(0.01, 0.98, interpretation_text, fontsize=8, va='top', ha='left', wrap=True) # Reduced font size
        ax_2_3.set_title("Interpretation & Bayesian Note", fontsize=10)

        fig.subplots_adjust(hspace=0.7, wspace=0.4, top=0.93, bottom=0.12, left=0.06, right=0.97) # Adjusted spacing
        plt.show()

    print("--- Finished generating all plots. ---")



def plot_y_bins_rolling_histogram(
    y_target_df,
    bin_edges,
    bin_centers,
    window_step=7,
    first_n_rows=150,
    title="Rolling Histogram of Next-30d Return Distribution",
    figsize_base=12,
    width_scale=1.8,
    bar_alpha=0.7,
    vmin=-0.1,
    vmax=0.1
):
    """
    Plot a rolling histogram and expected value summary from binned target returns.
    """
    # Calculate bin centers
    bin_centers = []
    for i in range(len(bin_edges)-1):
        if np.isinf(bin_edges[i]):
            left = bin_edges[i+1] - 0.01
        else:
            left = bin_edges[i]
        if np.isinf(bin_edges[i+1]):
            right = bin_edges[i] + 0.01
        else:
            right = bin_edges[i+1]
        bin_centers.append(0.5 * (left + right))
    
    step = window_step
    y_target_short = y_target_df.iloc[:first_n_rows]
    dates = y_target_short.index
    window_dates = [str(dates[i].date()) for i in range(0, len(dates), step)]
    
    histograms = []
    expected_vals = []
    for i in range(0, len(y_target_short), step):
        window = y_target_short.iloc[i:i+step]
        hist = window.mean(axis=0)
        histograms.append(hist.values)
        exp_val = np.dot(hist.values, bin_centers)
        expected_vals.append(exp_val)
    histograms = np.array(histograms)
    expected_vals = np.array(expected_vals)
    
    num_windows = histograms.shape[0]
    x = np.arange(num_windows)
    
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(figsize_base, num_windows//2), 5), 
        gridspec_kw={"height_ratios": [4, 0.7]}, sharex=True
    )
    
    # Histogram row (top)
    colors = []
    for c in bin_centers:
        if c < 0:
            colors.append('firebrick')
        elif c == 0:
            colors.append('gray')
        else:
            colors.append('forestgreen')
    
    bar_height = (bin_centers[1] - bin_centers[0]) * 0.55
    
    for i in range(num_windows):
        ax1.barh(
            y=bin_centers,
            width=histograms[i, :] * width_scale,
            left=i,
            height=bar_height,
            color=colors,
            alpha=bar_alpha,
            edgecolor='k'
        )
    
    ax1.set_ylabel('Bin center: next-30d return (%)')
    ax1.set_title(title)
    ax1.set_yticks(bin_centers)
    ax1.set_yticklabels([f"{c*100:.1f}%" for c in bin_centers])
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.2, alpha=0.8, zorder=1)
    ax1.set_xlim(-0.5, num_windows - 0.5)
    
    # Expected value row (bottom)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = plt.cm.get_cmap('RdYlGn')  # Red-Yellow-Green
    
    for i in range(num_windows):
        ev = expected_vals[i]
        color = cmap(norm(ev)) if not np.isnan(ev) else 'lightgray'
        rect = ax2.bar(i, 1, color=color, width=1.0, edgecolor='none')
        if not np.isnan(ev):
            ax2.text(i, 0.5, f"{ev*100:.2f}%", va='center', ha='center', fontsize=9, color='black')
    
    ax2.set_yticks([])
    ax2.set_ylabel("E[ret]")
    ax2.set_xlim(-0.5, num_windows - 0.5)
    ax2.set_ylim(0, 1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(window_dates, rotation=90, fontsize=9)
    ax2.set_xlabel('Date (window start)')
    
    # Add colorbar as legend for expected value
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='2.5%', pad=0.15)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label('Expected Value (% return)')
    cbar.set_ticks([vmin, 0, vmax])
    cbar.set_ticklabels([f"{vmin*100:.1f}%", "0%", f"{vmax*100:.1f}%"])
    
    plt.tight_layout()
    plt.show()



def plot_ev_by_cluster_scatter(y_target_df_with_ev, chart_title: str, jitter_strength=0.1):
    """
    Visualizes EV distribution per cluster.
    """
    if not all(col in y_target_df_with_ev.columns for col in ['EV', 'Cluster_Assignment']):
        print("Error: 'EV' and/or 'Cluster_Assignment' column(s) not found in y_target_df_with_ev.")
        print("Please ensure these columns are present.")
        return
    df_plot = y_target_df_with_ev.copy()
    ev_values = df_plot['EV']
    # Define EV thresholds and colors (same as before)
    conditions = [
        ev_values.isna(),
        ev_values < -0.05,
        (ev_values >= -0.05) & (ev_values < -0.01),
        (ev_values >= -0.01) & (ev_values <= 0.01),
        (ev_values > 0.01) & (ev_values <= 0.05),
        ev_values > 0.05
    ]
    color_map_names = ['grey', 'darkred', 'lightcoral', 'yellow', 'lightgreen', 'darkgreen']
    df_plot['EV_Color'] = np.select(conditions, color_map_names, default='black')
    plt.figure(figsize=(12, 4))
    unique_clusters = sorted(df_plot['Cluster_Assignment'].dropna().unique())

    # Define color labels for the legend
    legend_labels = {
        'darkred': 'Deep Red (EV < -0.05)',
        'lightcoral': 'Light Red (-0.05 <= EV < -0.01)',
        'yellow': 'Yellow (-0.01 <= EV <= 0.01)',
        'lightgreen': 'Light Green (0.01 < EV <= 0.05)',
        'darkgreen': 'Deep Green (EV > 0.05)',
        'grey': 'NaN EV'
    }
    # Plot points for each EV color category to build the legend correctly
    for color_hex, label in legend_labels.items():
        subset = df_plot[df_plot['EV_Color'] == color_hex]
        if not subset.empty:
            # Apply jitter to x-coordinates
            x_jittered = subset['Cluster_Assignment'] + np.random.normal(0, jitter_strength, size=len(subset))
            plt.scatter(x_jittered, subset['EV'], label=label, color=color_hex, alpha=0.7, s=50)
    plt.title(chart_title, fontsize=16)
    plt.xlabel('Cluster Assignment', fontsize=14)
    plt.ylabel('Expected Returns ', fontsize=14)
    
    # Set x-axis ticks to be the actual cluster labels
    if unique_clusters: # Check if unique_clusters is not empty
        plt.xticks(ticks=unique_clusters, labels=[f'Cluster {c}' for c in unique_clusters])
    else:
        plt.xticks([]) # No ticks if no clusters

    plt.grid(True, linestyle='--', alpha=0.7, axis='y') # Grid on y-axis can be helpful
    # plt.legend(title='Returns Category')
    plt.tight_layout()
    plt.show()








############




def plot_mean_ev_category_shift_comparison(hypothesis_df: pd.DataFrame, chart_title: str):
    """
    Plots bar charts comparing Mean EV for train vs. validation,
    annotated with the Mean EV category shift information.

    Args:
        hypothesis_df (pd.DataFrame): Output from analyze_cluster_mean_ev_category_shift.
                                      Expected columns: 'Mean_EV_Train', 'Mean_EV_Val',
                                                      'Mean_EV_Category_Shift',
                                                      'Category_Train (MeanEV)',
                                                      'Category_Val (MeanEV)'.
    """
    if hypothesis_df.empty:
        print("Hypothesis DataFrame for category shifts is empty. Nothing to plot.")
        return

    metrics_to_plot = [
        (chart_title, 'Mean_EV_Train', 'Mean_EV_Val')
    ]
    
    n_metrics = len(metrics_to_plot) # This will be 1
    clusters = hypothesis_df.index.map(str).tolist()
    x = np.arange(len(clusters)) # x locations for the groups
    width = 0.35 # width of the bars

    # Since n_metrics is 1, we'll have one subplot.
    # Adjust figsize if it's just one plot.
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(clusters)), 4)) # Dynamic width

    plot_label, col_train, col_val = metrics_to_plot[0] # Get the single metric
    
    if col_train not in hypothesis_df.columns or col_val not in hypothesis_df.columns:
        ax.text(0.5, 0.5, f"Required columns\n'{col_train}' or '{col_val}'\nnot found in DataFrame.",
                ha='center', va='center', transform=ax.transAxes, color='grey', fontsize=10)
        ax.set_title(plot_label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(clusters, rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
        return

    train_values = hypothesis_df[col_train].fillna(0)
    val_values = hypothesis_df[col_val].fillna(0)

    ax.bar(x - width/2, train_values, width, label='Train Mean EV', alpha=0.7, color='cornflowerblue')
    ax.bar(x + width/2, val_values, width, label='Validation Mean EV', alpha=0.7, color='lightcoral')

    # Annotations based on Mean_EV_Category_Shift
    for j, cluster_label in enumerate(clusters):
        category_shift_interp = hypothesis_df['Mean_EV_Category_Shift'].iloc[j]
        cat_train_label = hypothesis_df['Category_Train (MeanEV)'].iloc[j]
        cat_val_label = hypothesis_df['Category_Val (MeanEV)'].iloc[j]

        display_text = ""
        text_color = 'grey' # Default

        if category_shift_interp == "Yes":
            display_text = f"Category Shift!\n({cat_train_label} → {cat_val_label})"
            text_color = 'red'
        elif category_shift_interp == "No":
            display_text = f"Category Stable\n({cat_train_label})"
            text_color = 'green'
        elif "N/A" in category_shift_interp:
            display_text = "Shift N/A\n(NaN EV)"
            text_color = 'orange'
        
        # Optional: Add Chi2 internal shift info if available and desired
        chi2_interp = hypothesis_df.get('Chi2_Interpretation (Internal)', pd.Series(dtype=str)).iloc[j]
        if pd.notna(chi2_interp) and "shifted" in str(chi2_interp).lower() and display_text:
            display_text += f"\nInternal: {chi2_interp.split('(')[0].strip()}" # Shorten internal message
            if text_color == 'green': # If mean category was stable, but internal shifted
                 text_color = 'darkgoldenrod'


        y_pos_text = max(train_values.iloc[j], val_values.iloc[j], 0) # Ensure y_pos is not below 0 for text base
        min_val_text = min(train_values.iloc[j], val_values.iloc[j], 0)

        # Position text: if bars are very different, place near taller; if similar, centered above.
        # More robust positioning
        text_y_base = y_pos_text
        va_text = 'bottom'
        if abs(y_pos_text - min_val_text) < 0.1 * abs(y_pos_text) and y_pos_text < 0 : # both negative and close
             text_y_base = min_val_text
             va_text = 'top'
        elif y_pos_text < 0: # only max is negative
             va_text = 'top'


        offset_factor = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]) # 5% of y-axis range
        text_y = text_y_base + offset_factor if va_text == 'bottom' else text_y_base - offset_factor


        ax.text(x[j], text_y, display_text,
                ha='center', va=va_text,
                fontsize=8, color=text_color, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7) if display_text else None)


    ax.set_ylabel('Mean EV')
    ax.set_title(plot_label, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(clusters, rotation=45, ha="right")
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    fig.suptitle('Mean EV & Profit Category Shift (Train vs Validation)', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



def _categorize_ev_for_cluster_shift_hypo_test(ev_value: float, thresholds: dict) -> str:
    """
    Categorizes an EV value into predefined profit/loss categories.
    """
    if pd.isna(ev_value):
        return "Unknown (NaN EV)"
    if ev_value < thresholds['deep_loss_upper']: return "Deep Loss"
    if ev_value < thresholds['loss_upper']: return "Loss"
    if ev_value <= thresholds['neutral_upper']: return "Neutral"
    if ev_value <= thresholds['profit_upper']: return "Profit"
    return "Deep Profit"

def analyze_cluster_ev_category_shifts(
    y_train_processed: pd.DataFrame,
    y_val_processed: pd.DataFrame,
    ev_category_thresholds: dict,
    test_internal_distribution: bool = False # If True, also run Chi2 test on internal EV categories
) -> pd.DataFrame:
    """
    Analyzes if the mean EV of clusters shifts between predefined profit/loss categories
    from train to validation. Optionally tests for shifts in the internal distribution
    of EV values across these categories within each cluster.
    """
    if not all(col in y_train_processed.columns for col in ['Cluster_Assignment', 'EV']):
        raise ValueError("y_train_processed must contain 'Cluster_Assignment' and 'EV' columns.")
    if not all(col in y_val_processed.columns for col in ['Cluster_Assignment', 'EV']):
        raise ValueError("y_val_processed must contain 'Cluster_Assignment' and 'EV' columns.")

    train_clusters = set(y_train_processed['Cluster_Assignment'].unique())
    val_clusters = set(y_val_processed['Cluster_Assignment'].unique())
    common_clusters = sorted(list(train_clusters.intersection(val_clusters)))
    
    category_shift_results = []
    category_names = ["Deep Loss", "Loss", "Neutral", "Profit", "Deep Profit", "Unknown (NaN EV)"]


    for cluster_id in common_clusters:
        ev_train_all = y_train_processed.loc[y_train_processed['Cluster_Assignment'] == cluster_id, 'EV'] # Keep NaNs for now if any
        ev_val_all = y_val_processed.loc[y_val_processed['Cluster_Assignment'] == cluster_id, 'EV']

        n_train = len(ev_train_all.dropna()) # Count non-NaN for N
        n_val = len(ev_val_all.dropna())

        # Calculate mean EV for categorization (handle empty series)
        mean_ev_train = ev_train_all.dropna().mean() if n_train > 0 else np.nan
        mean_ev_val = ev_val_all.dropna().mean() if n_val > 0 else np.nan
        
        category_train = _categorize_ev_for_cluster_shift_hypo_test(mean_ev_train, ev_category_thresholds)
        category_val = _categorize_ev_for_cluster_shift_hypo_test(mean_ev_val, ev_category_thresholds)
        
        mean_ev_category_shifted = "Yes" if category_train != category_val and category_train != "Unknown (NaN EV)" and category_val != "Unknown (NaN EV)" else "No"
        if category_train == "Unknown (NaN EV)" or category_val == "Unknown (NaN EV)":
             mean_ev_category_shifted = "N/A (due to NaN mean EV)"


        result_dict = {
            'Cluster': cluster_id,
            'N_train': n_train,
            'N_val': n_val,
            'Mean_EV_Train': mean_ev_train,
            'Mean_EV_Val': mean_ev_val,
            'Category_Train (MeanEV)': category_train,
            'Category_Val (MeanEV)': category_val,
            'Mean_EV_Category_Shift': mean_ev_category_shifted,
            'Chi2_p_value (Internal)': np.nan,
            'Chi2_Interpretation (Internal)': 'Not tested'
        }

        if test_internal_distribution and n_train >= 5 and n_val >= 5:
            train_cats = ev_train_all.apply(lambda x: categorize_ev(x, ev_category_thresholds)).value_counts().reindex(category_names, fill_value=0)
            val_cats = ev_val_all.apply(lambda x: categorize_ev(x, ev_category_thresholds)).value_counts().reindex(category_names, fill_value=0)
            
            contingency_table = pd.DataFrame({'Train': train_cats, 'Val': val_cats})
            
            # Remove categories where both train and val have zero counts to avoid issues with chi2
            contingency_table = contingency_table.loc[(contingency_table.sum(axis=1) > 0)]

            if contingency_table.shape[0] >= 2 and contingency_table.shape[1] == 2: # Need at least 2 categories with some data
                try:
                    chi2_stat, chi2_pval, _, _ = chi2_contingency(contingency_table)
                    result_dict['Chi2_p_value (Internal)'] = chi2_pval
                    if pd.isna(chi2_pval): result_dict['Chi2_Interpretation (Internal)'] = "Chi2 error"
                    elif chi2_pval <= 0.05: result_dict['Chi2_Interpretation (Internal)'] = "Internal category distribution likely shifted"
                    else: result_dict['Chi2_Interpretation (Internal)'] = "Internal category distribution likely similar"
                except ValueError: # e.g. if table sums to zero
                     result_dict['Chi2_Interpretation (Internal)'] = "Chi2 error (e.g. low counts)"
            else:
                result_dict['Chi2_Interpretation (Internal)'] = "Insufficient distinct categories for Chi2"
        
        category_shift_results.append(result_dict)

    shift_df = pd.DataFrame(category_shift_results)
    if not shift_df.empty:
        shift_df = shift_df.drop_duplicates(subset=['Cluster']).set_index('Cluster')
    return shift_df
