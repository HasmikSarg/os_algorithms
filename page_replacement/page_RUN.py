"""
Page replacement algorithm simulation for page sets.
    Runs tests on data from input_data/ and exports results to output_data/
    Analyzes and creates performance summaries in .csv, performance charts in .png, and final reports in .txt
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional
from datetime import datetime

from page_simulate import PageReplacementSimulator
from page_generate import PageGenerator, Pages


class ReplacementAnalysis(PageReplacementSimulator, PageGenerator, Pages):
    """
    Analyze page replacement algorithms.
    """
    
    def __init__(self):
        super().__init__()

        self.generator = PageGenerator()
        self.simulator = PageReplacementSimulator()

        self.input_directory = os.path.join(os.path.dirname(__file__), 'input_data')
        self.output_directory = os.path.join(os.path.dirname(__file__), 'output_data')
        os.makedirs(self.input_directory, exist_ok=True)
        os.makedirs(self.output_directory, exist_ok=True)

        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        self.color_scheme = {
            'hit_rate': '#2ecc71',
            'page_faults': '#e74c3c',
            'execution_time': '#f39c12',
            'histogram': '#9b59b6',
            'optimal': '#27ae60',
            'mean_line': '#c0392b',
            'median_line': '#d35400',
            'interval_bars': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        }

        self.algorithm_colors = {
            'LRU': '#3498db',
            'LFU': '#e67e22',
        }

        self._load_algorithms()
        
    def create_charts(self, results: List, dataset_name: str):
        """Create charts for dataset"""

        output_dataset_dir = os.path.join(self.output_directory, dataset_name)
        os.makedirs(output_dataset_dir, exist_ok=True)
        
        df_results = self.simulator.results_to_dataframe(results)
        
        df_chart = df_results.copy()
        df_chart['algorithm_name'] = df_chart.apply(
            lambda row: f"{row['algorithm']} (q={int(row['quantum'])})" 
            if row['algorithm'] == 'RR' and pd.notna(row['quantum']) 
            else row['algorithm'], axis=1
        )
        
        self._create_performance_charts(results, dataset_name)
        self._create_frame_analysis(df_results, dataset_name)

        df_results.to_csv(os.path.join(output_dataset_dir, f'{dataset_name}_results.csv'), index=False)

    def _create_performance_charts(self, results: List, dataset_name: str):
        """Charts for PH, PF, HR, FR for each algorithm"""

        output_dataset_dir = os.path.join(self.output_directory, dataset_name)
        os.makedirs(output_dataset_dir, exist_ok=True)
        df_results = self.simulator.results_to_dataframe(results)
        if df_results.empty:
            print(f"No data for dataset {dataset_name}")
            return
        chart_data = df_results.groupby('algorithm', observed=False).agg({
            'hit_percentage': 'mean',
            'page_faults': 'mean',
            'execution_time': 'mean'
        }).reset_index()
        plt.figure(figsize=(15, 10), constrained_layout=True)

        plt.subplot(2, 2, 1)
        for algorithm in df_results['algorithm'].unique():
            alg_data = df_results[df_results['algorithm'] == algorithm]
            mean_data = alg_data.groupby('frame_count', observed=False)['hit_percentage'].mean().reset_index()
            color = self.algorithm_colors.get(algorithm, self.color_scheme['hit_rate'])
            plt.plot(mean_data['frame_count'], mean_data['hit_percentage'], 
                    marker='o', label=algorithm, linewidth=2, markersize=6, color=color)
        plt.xlabel('Frame count')
        plt.ylabel('Hit rate (%)')
        plt.title(f'Hit rate', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        for algorithm in df_results['algorithm'].unique():
            alg_data = df_results[df_results['algorithm'] == algorithm]
            mean_data = alg_data.groupby('frame_count', observed=False)['page_faults'].mean().reset_index()
            color = self.algorithm_colors.get(algorithm, self.color_scheme['page_faults'])
            plt.plot(mean_data['frame_count'], mean_data['page_faults'], 
                    marker='s', label=algorithm, linewidth=2, markersize=6, color=color)
        plt.xlabel('Frame count')
        plt.ylabel('Page faults count')
        plt.title(f'Page faults', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        colors = [self.algorithm_colors.get(alg, self.color_scheme['hit_rate']) for alg in chart_data['algorithm']]
        bars = plt.bar(chart_data['algorithm'], chart_data['hit_percentage'], 
                      color=colors, alpha=0.8)
        plt.xlabel('Algorithm')
        plt.ylabel('Average hit rate (%)')
        plt.title(f'Average hit rate', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        interval_stats = self._calculate_fault_intervals(df_results, interval_count=7)
        if not interval_stats.empty:
            algorithms = interval_stats['algorithm'].unique()
            intervals = interval_stats['fault_interval'].unique()
            intervals = [x for x in intervals if pd.notna(x)]
            if len(intervals) > 0 and len(algorithms) > 0:
                x_pos = np.arange(len(intervals))
                width = 0.35
                for i, algorithm in enumerate(algorithms):
                    alg_data = interval_stats[interval_stats['algorithm'] == algorithm]
                    counts = []
                    for interval in intervals:
                        interval_count = alg_data[alg_data['fault_interval'] == interval]['count'].sum()
                        counts.append(interval_count)
                    positions = x_pos + (i - len(algorithms)/2 + 0.5) * width / len(algorithms)
                    color = self.algorithm_colors.get(algorithm, self.color_scheme['interval_bars'][i % len(self.color_scheme['interval_bars'])])
                    plt.bar(positions, counts, width/len(algorithms), 
                           label=algorithm, alpha=0.8, color=color)
                plt.xlabel('Page fault intervals')
                plt.ylabel('Case count')
                plt.title(f'Page fault distribution', fontsize=12, fontweight='bold')
                plt.xticks(x_pos, intervals, rotation=45)
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No data\nfor intervals', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes, fontsize=12)
                plt.title(f'Page fault intervals - {dataset_name}')
        else:
            plt.text(0.5, 0.5, 'No data\nfor page fault intervals', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title(f'Page fault intervals - {dataset_name}')
        plt.suptitle(f'Performance analysis - {dataset_name}', fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(output_dataset_dir, f'{dataset_name}_performance_overview.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _create_frame_analysis(self, df_results: pd.DataFrame, dataset_name: str):
        """Detailed frame analysis"""

        output_dataset_dir = os.path.join(self.output_directory, dataset_name)
        plt.figure(figsize=(10, 15), constrained_layout=True)

        plt.subplot(2, 1, 1)
        for algorithm in df_results['algorithm'].unique():
            alg_data = df_results[df_results['algorithm'] == algorithm]
            mean_data = alg_data.groupby('frame_count', observed=False)['execution_time'].mean().reset_index()
            color = self.algorithm_colors.get(algorithm, self.color_scheme['execution_time'])
            plt.plot(mean_data['frame_count'], mean_data['execution_time'], marker='o', label=algorithm, linewidth=2, markersize=6, color=color)
        plt.xlabel('Frame count')
        plt.ylabel('Average execution time (s)')
        plt.title(f'Algorithm execution time vs frame count', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        for algorithm in df_results['algorithm'].unique():
            alg_data = df_results[df_results['algorithm'] == algorithm]
            color = self.algorithm_colors.get(algorithm, self.color_scheme['execution_time'])
            plt.plot(alg_data['page_faults'], alg_data['execution_time'], marker='o', label=algorithm, linewidth=2, markersize=6, color=color)
        plt.xlabel('Page fault count')
        plt.ylabel('Execution time (s)')
        plt.title(f'Execution time vs. page fault count', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.suptitle(f'Execution time analysis - {dataset_name}', fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(output_dataset_dir, f'{dataset_name}_time_vs_faults.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_summary_charts(self, df_results: pd.DataFrame):
        """
        Create summary charts of page replacement algorithm performance
        """

        print("\n>>>>> creating charts <<<<<")
        if df_results.empty:
            print("No data to create charts")
            return
        plt.figure(figsize=(16, 12))

        plt.subplot(3, 2, 1)
        for algorithm in df_results['algorithm'].unique():
            alg_data = df_results[df_results['algorithm'] == algorithm]
            mean_data = alg_data.groupby('frame_count', observed=False)['hit_percentage'].mean().reset_index()
            color = self.algorithm_colors.get(algorithm, self.color_scheme['hit_rate'])
            plt.plot(mean_data['frame_count'], mean_data['hit_percentage'], 
                    marker='o', label=algorithm, linewidth=2, color=color)
        plt.xlabel('Frame count')
        plt.ylabel('Average hit rate (%)')
        plt.title('Algorithm performance vs frame count')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 2, 2)
        pivot_data = df_results.groupby(['algorithm', 'sequence_type'], observed=False)['hit_percentage'].mean().reset_index()
        if not pivot_data.empty:
            pivot_table = pivot_data.pivot(index='algorithm', columns='sequence_type', values='hit_percentage')
            pivot_table.plot(kind='bar', ax=plt.gca(), alpha=0.8, 
                           color=[self.algorithm_colors.get(alg, self.color_scheme['hit_rate']) for alg in pivot_table.index])
            plt.xlabel('Algorithm')
            plt.ylabel('Average hit rate (%)')
            plt.title('Performance by sequence type')
            plt.xticks(rotation=45)
            plt.legend(title='Sequence type', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.subplot(3, 2, 3)
        interval_stats = self._calculate_fault_intervals(df_results, interval_count=7)
        if not interval_stats.empty:
            algorithms = interval_stats['algorithm'].unique()
            intervals = interval_stats['fault_interval'].unique()
            intervals = [x for x in intervals if pd.notna(x)]
            if len(intervals) > 0 and len(algorithms) > 0:
                x_pos = np.arange(len(intervals))
                width = 0.35
                for i, algorithm in enumerate(algorithms):
                    alg_data = interval_stats[interval_stats['algorithm'] == algorithm]
                    counts = []
                    for interval in intervals:
                        interval_count = alg_data[alg_data['fault_interval'] == interval]['count'].sum()
                        counts.append(interval_count)
                    positions = x_pos + (i - len(algorithms)/2 + 0.5) * width / len(algorithms)
                    color = self.algorithm_colors.get(algorithm, self.color_scheme['interval_bars'][i % len(self.color_scheme['interval_bars'])])
                    plt.bar(positions, counts, width/len(algorithms), 
                           label=algorithm, alpha=0.8, color=color)
                plt.xlabel('Page fault intervals')
                plt.ylabel('Case count')
                plt.title('Global page fault intervals')
                plt.xticks(x_pos, intervals, rotation=45)
                plt.legend()
                plt.grid(True, alpha=0.3)

        plt.subplot(3, 2, 4)
        for algorithm in df_results['algorithm'].unique():
            alg_data = df_results[df_results['algorithm'] == algorithm]
            mean_data = alg_data.groupby('frame_count', observed=False)['page_faults'].mean().reset_index()
            color = self.algorithm_colors.get(algorithm, self.color_scheme['page_faults'])
            plt.plot(mean_data['frame_count'], mean_data['page_faults'], 
                    marker='s', label=algorithm, linewidth=2, color=color)
        plt.xlabel('Frame count')
        plt.ylabel('Average page fault count')
        plt.title('Page faults vs frame count')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 2, 5)
        execution_data = df_results.groupby('algorithm', observed=False)['execution_time'].mean().reset_index()
        colors = [self.algorithm_colors.get(alg, self.color_scheme['execution_time']) for alg in execution_data['algorithm']]
        bars = plt.bar(execution_data['algorithm'], execution_data['execution_time'], 
                      alpha=0.8, color=colors)
        plt.xlabel('Algorithm')
        plt.ylabel('Average execution time (ms)')
        plt.title('Algorithm execution time')
        plt.xticks(rotation=45)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.8f}', ha='center', va='bottom', fontsize=9)
            
        plt.subplot(3, 2, 6)
        summary_data = df_results.groupby('algorithm', observed=False).agg({
            'hit_percentage': 'mean',
            'page_faults': 'mean'
        }).reset_index()
        x = range(len(summary_data))
        width = 0.35
        ax1 = plt.gca()
        bars1 = ax1.bar([i - width/2 for i in x], summary_data['hit_percentage'], 
                       width, label='Hit rate (%)', alpha=0.8, color=self.color_scheme['hit_rate'])
        ax1.set_ylabel('Hit rate (%)', color=self.color_scheme['hit_rate'])
        ax1.tick_params(axis='y', labelcolor=self.color_scheme['hit_rate'])
        ax2 = ax1.twinx()
        bars2 = ax2.bar([i + width/2 for i in x], summary_data['page_faults'], 
                       width, label='Page faults', alpha=0.8, color=self.color_scheme['page_faults'])
        ax2.set_ylabel('Page faults', color=self.color_scheme['page_faults'])
        ax2.tick_params(axis='y', labelcolor=self.color_scheme['page_faults'])
        plt.title('Performance summary')
        ax1.set_xticks(x)
        ax1.set_xticklabels(summary_data['algorithm'], rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_directory, 'page_replacement_performance_overview.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    
    def _calculate_fault_intervals(self, df_results: pd.DataFrame, interval_count: int = 7):
        """
        Calculate page fault intervals
        """

        if df_results.empty:
            return pd.DataFrame()
        
        min_faults = df_results['page_faults'].min()
        max_faults = df_results['page_faults'].max()

        interval_edges = np.linspace(min_faults, max_faults + 1, interval_count + 1)
        interval_labels = []
        for i in range(interval_count):
            start = int(interval_edges[i])
            end = int(interval_edges[i + 1]) - 1
            if i == interval_count - 1:
                end = int(interval_edges[i + 1])
            interval_labels.append(f"{start}-{end}")

        df_results['fault_interval'] = pd.cut(
            df_results['page_faults'], 
            bins=interval_edges, 
            labels=interval_labels, 
            include_lowest=True,
            right=False
        )

        interval_stats = df_results.groupby(['algorithm', 'fault_interval'], observed=False).agg({
            'page_faults': ['mean', 'count'],
            'hit_percentage': 'mean'
        }).round(4)
        interval_stats.columns = ['mean_faults', 'count', 'mean_hit_rate']
        interval_stats = interval_stats.reset_index()
        return interval_stats

    def run_comparison(self, algorithms: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare datasets with charts and reports
        """

        print("\n>>>>> running comparison <<<<<")
        input_datasets = self.show_datasets()
        if not input_datasets:
            print("No datasets in input directory")
            self.generator.generate_datasets()
            input_datasets = self.show_datasets()

        if algorithms is None:
            algorithms = list(self.simulator.algorithms.keys())

        all_results = []

        for dataset_name in input_datasets:
            print(f"Processing dataset: {dataset_name}")
            sequence = self.load_dataset(dataset_name)
            dataset_type = dataset_name.split('_')[0] if '_' in dataset_name else 'unknown'
            dataset_results = []
            for algorithm in algorithms:
                if algorithm in self.simulator.algorithms:
                    results = self.simulator.run_dataset_test(
                        algorithms=[algorithm],
                        frame_counts=self.frame_counts,
                        sequence=sequence,
                        sequence_type=dataset_type
                    )
                    dataset_results.extend(results)
                    all_results.extend(results)
            if dataset_results:
                self.create_charts(dataset_results, dataset_name)
                df_dataset = self.simulator.results_to_dataframe(dataset_results)
                dataset_report = self.create_report(df_dataset, title=f"DATASET REPORT: {dataset_name}")
                output_dataset_dir = os.path.join(self.output_directory, dataset_name)
                os.makedirs(output_dataset_dir, exist_ok=True)
                report_file = os.path.join(output_dataset_dir, f"{dataset_name}_report.txt")
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(dataset_report)

        df_results = self.simulator.results_to_dataframe(all_results)
        output_file = os.path.join(self.output_directory, 'page_replacement_results.csv')
        df_results.to_csv(output_file, index=False)

        if 'sequence_type' in df_results.columns:
            for dataset_type in sorted(df_results['sequence_type'].unique()):
                type_data = df_results[df_results['sequence_type'] == dataset_type]
                if type_data.empty:
                    continue
                type_report = self.create_report(type_data, title=f"DATASET TYPE REPORT: {dataset_type}")
                type_report_file = os.path.join(self.output_directory, f"{dataset_type}_report.txt")
                with open(type_report_file, 'w', encoding='utf-8') as f:
                    f.write(type_report)

        self.create_summary_charts(df_results)
        return df_results


    def create_report(self, df_results: pd.DataFrame, title: str = "REPORT") -> str:
        """
        Generate text report.
        """
        
        if df_results.empty:
            return f"{title}\nNo data for report."
        report_content = []

        report_content = [
            f" {'_'*(len(title)+8)} ",
            f"|{' '*(len(title)+8)}|",
            f"|    {title}    |",
            f"|{'_'*(len(title)+8)}|",
            "",
            ">>>>>> LEGEND <<<<<<",
            "\tPA (page accesses): page access requests",
            "\tPH (page hits): page hits",
            "\tPF (page faults): page faults",
            "\tHR (hit rate): hit rate = (PH / PA) * 100%",
            "\tFR (fault rate): fault rate = (PF / PA) * 100%",
            "\tRT (response time): average response time",
            "",
            ">>>>>> PARAMETERS <<<<<<",
            f"\tTest count: {len(df_results)}",
            f"\tTested algorithms: {sorted(df_results['algorithm'].unique())}",
            f"\tDataset types: {sorted(df_results['sequence_type'].unique())}",
            "",
        ]
        
        if not df_results.empty:
            report_content.extend([
                ">>>>>> STATISTICS <<<<<<",
            ])
            
            algorithm_analysis = df_results.groupby('algorithm', observed=False).agg({
                'page_faults': ['mean', 'min', 'max'],
                'hit_percentage': ['mean', 'min', 'max'],
                'execution_time': ['mean', 'min', 'max'],
            }).round(8)
            
            algorithm_analysis.columns = [f'{metric}_{stat}' for metric, stat in algorithm_analysis.columns]
            
            header_line1 = f"\t{'':<16}|{'page faults':<24}{' ':<4}{'hit rate':<24}"
            header_line2 = f"\t{'algorithm':<8}{'':<8}|{'mean':<8}{'min':<8}{'max':<8}{' ':<4}{'mean':<8}{'min':<8}{'max':<8}"
            separator_line = f"\t{'-'*8}{'-'*8}|{'-'*8*7}"
            report_content.extend([
                header_line1,
                header_line2,
                separator_line,
            ])
            
            faults_mean = []
            hits_mean = []
            execution_mean = []
            for algorithm in algorithm_analysis.index:
                algorithm_data = algorithm_analysis.loc[algorithm]

                faults_mean_val = f"{algorithm_data['page_faults_mean']:<8}"
                faults_min = f"{algorithm_data['page_faults_min']:<8}"
                faults_max = f"{algorithm_data['page_faults_max']:<4}"
                
                hits_mean_val = f"{algorithm_data['hit_percentage_mean']:<8}"
                hits_min = f"{algorithm_data['hit_percentage_min']:<8}"
                hits_max = f"{algorithm_data['hit_percentage_max']:<4}"

                faults_mean += [algorithm_data['page_faults_mean']]
                hits_mean += [algorithm_data['hit_percentage_mean']]
                execution_mean += [algorithm_data['execution_time_mean']]
                
                algorithm_line = f"{algorithm:<16}|{faults_mean_val}{faults_min}{faults_max}{' ':<8}{hits_mean_val}{hits_min}{hits_max}"
                report_content.append(f"\t{algorithm_line}")
            
            report_content.append("")
        
        if not df_results.empty:
            best_hits = df_results.loc[df_results['hit_percentage'].idxmax()]
            least_faults = df_results.loc[df_results['page_faults'].idxmin()]
            fastest = df_results.loc[df_results['execution_time'].idxmin()]
            report_content.extend([
                ">>>>>> RESULTS <<<<<<",
                f"\tHighest AHR for: {best_hits['algorithm']} = {best_hits['hit_percentage']:.2f}%",
                f"\tLowest APF for: {least_faults['algorithm']} = {least_faults['page_faults']:.0f}",
                f"\tShortest ART for: {fastest['algorithm']} = {fastest['execution_time']:.6f}s",
                "",
            ])
        report_content.extend([
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ])
        return "\n".join(report_content)

    def generate_analysis(self):
        """Generate analysis with charts and reports"""
        
        df_results = self.run_comparison()
        print("\n>>>>> creating report <<<<<")
        final_report = self.create_report(df_results, title="PAGE REPLACEMENT ALGORITHM ANALYSIS")
        final_report_file = os.path.join(self.output_directory, "page_replacement_report.txt")
        with open(final_report_file, "w", encoding="utf-8") as f:
            f.write(final_report)
            
        print("\nOutput files:")
        for file in sorted(os.listdir(self.output_directory)):
            if os.path.isfile(os.path.join(self.output_directory, file)):
                print(f"\t{file}")
            elif os.path.isdir(os.path.join(self.output_directory, file)):
                print(f"\t{file}/ (dataset charts folder)")

        return df_results


if __name__ == "__main__":
    banner=r'''

 ___                   ___            _                                   _      ___  _                 _    _    _              
| . \ ___  ___  ___   | . \ ___  ___ | | ___  ___  ___ ._ _ _  ___ ._ _ _| |_   | . || | ___  ___  _ _ <_> _| |_ | |_ ._ _ _  ___
|  _/<_> |/ . |/ ._>  |   // ._>| . \| |<_> |/ | '/ ._>| ' ' |/ ._>| ' | | |    |   || |/ . |/ . \| '_>| |  | |  | . || ' ' |<_-<
|_|  <___|\_. |\___.  |_\_\\___.|  _/|_|<___|\_|_.\___.|_|_|_|\___.|_|_| |_|    |_|_||_|\_. |\___/|_|  |_|  |_|  |_|_||_|_|_|/__/
          <___'                 |_|                                                     <___'                                    
'''

    print(banner)
    simulation = ReplacementAnalysis()
    title = "Analyzing page replacement algorithms"
    print(f'\n{"=" * len(title)}')
    print(title)
    print(f'{"=" * len(title)}\n')

    print(f"\t{', '.join(simulation.simulator._load_algorithms())}")
    print(f"\t{', '.join(simulation.simulator.show_datasets())}")

    simulation.generate_analysis()