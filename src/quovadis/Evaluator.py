import itertools
import os
from typing import List
import trackeval
import pandas as pd


class Evaluator:

    def __init__(self):

        self.default_eval_config = trackeval.Evaluator.get_default_eval_config()
        self.default_eval_config['DISPLAY_LESS_PROGRESS'] = False
        self.default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        self.default_metrics_config = {'METRICS': [
            'HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}

        self.results_df = []

    def _get_configs(self, dataset, tracker_names, sequence_names, tracker_folder, output_folder, output_sub_folder):
        config = {**self.default_eval_config, **self.default_dataset_config,
                  **self.default_metrics_config}  # Merge default configs
        config["SEQ_INFO"] = dict(
            zip(sequence_names, len(sequence_names)*[None]))
        config["GT_FOLDER"] = f'./data/{dataset}/sequences'

        config["TRACKERS_TO_EVAL"] = tracker_names
        config["BENCHMARK"] = dataset
        config["PRINT_RESULTS"] = True
        config["TRACKERS_FOLDER"] = tracker_folder
        config["OUTPUT_FOLDER"] = output_folder if output_folder else tracker_folder
        config['OUTPUT_SUB_FOLDER'] = output_sub_folder if output_sub_folder else ""
        print(config)
        eval_config = {k: v for k, v in config.items(
        ) if k in self.default_eval_config.keys()}
        dataset_config = {k: v for k, v in config.items(
        ) if k in self.default_dataset_config.keys()}
        metrics_config = {k: v for k, v in config.items(
        ) if k in self.default_metrics_config.keys()}
        return eval_config, dataset_config, metrics_config

    def eval(self, dataset: str, sequence_names: List[str], tracker_names: List[str], tracker_folder: str):

        # eval quo vadis output and baseline
        for data_folder, model_name, seq_names in [[f'./data/{dataset}/tracker', "Baseline", sequence_names],
                                                   [tracker_folder, "QuoVadis", sequence_names]]:
            print(tracker_folder)

            eval_config, dataset_config, metrics_config = self._get_configs(
                dataset, tracker_names,  sequence_names, data_folder, output_folder=tracker_folder, output_sub_folder=model_name.lower())
            # Run code
            evaluator = trackeval.Evaluator(eval_config)
            dataset_list = [
                trackeval.datasets.MotChallenge2DBox(dataset_config)]
            metrics_list = []
            for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
                if metric.get_name() in metrics_config['METRICS']:
                    metrics_list.append(metric(metrics_config))
            if len(metrics_list) == 0:
                raise Exception('No metrics selected for evaluation')
            evaluator.evaluate(dataset_list, metrics_list)
            for tracker in tracker_names:

                df = pd.read_csv(os.path.join(dataset_list[0].get_output_fol(
                    tracker), "pedestrian_summary.txt"), sep=' ')
                df["model"] = model_name
                df["seq"] = seq_names[0] if len(seq_names) == 1 else "combined"
                df["tracker"] = tracker
                self.results_df.append(df)

    def _collect_data(self, columns=[]):
        df = pd.concat(self.results_df)
        pd.set_option('display.max_columns', 500)
        if len(columns) > 0:
            columns = ["model", "tracker", "seq", ] + columns
            df = df[columns]
        return df

    def load_results(self, dataset, sequences,  trackers, add_baseline=True):
        print(dataset)
        data_folder = f'./run/output/{dataset}'
        models = ["quovadis"] + ["baseline"] if add_baseline else []
        for seq, tracker, model in itertools.product(sequences, trackers, models):

            df = pd.read_csv(os.path.join(data_folder, tracker,
                             "eval", model, seq, "pedestrian_summary.txt"), sep=' ')
            df["model"] = model
            df["seq"] = seq
            df["tracker"] = tracker
            self.results_df.append(df)
        self._collect_data()

    def print_results(self, columns=[]):
        df = self._collect_data(columns)
        print(df)

    def print_md_table(self, dataset, sequences, trackers,   columns=[], add_baseline=True):
        def _tab(tab_level):
            return "\t"*tab_level
        self.load_results(dataset, sequences, trackers,
                          add_baseline=add_baseline)
        df = self._collect_data(columns)

        df = df[df.seq == "combined"]

        if len(columns) == 0:
            columns = list(df.columns)
        mk_down_table = f'<center> \n{_tab(1)}<table> \n{_tab(2)}<thead> \n{_tab(3)}<tr>\n{_tab(4)}<th>{dataset}</th>'
        for tracker in trackers:
            mk_down_table += f'<th>{tracker}</th>'
        mk_down_table += f'\n{_tab(3)}</tr>\n{_tab(2)}<thead>\n{_tab(2)}<tbody>\n'
        for metric in columns:
            mk_down_table += f"{_tab(3)}<tr>\n{_tab(4)}<td><b>{metric}</b</td>"
            for tracker in trackers:
                mk_down_table += "<td>"
                value_quovadis = df.loc[((df.tracker == tracker) & (
                    df.model == "quovadis")), metric]

                if type(value_quovadis.item()) is float:
                    mk_down_table += f'{value_quovadis.item():.2f}'
                else:
                    mk_down_table += f'{value_quovadis.item()}'

                if add_baseline:
                    value_baseline = df.loc[((df.tracker == tracker) & (
                        df.model == "baseline")), metric]
                    difference = value_quovadis.item() - value_baseline.item()
                    if type(value_baseline.item()) is float:
                        mk_down_table += ' (' + ("+" if difference >
                                                0 else "") + f'{difference:.2f})'
                    else:
                        mk_down_table += ' (' + ("+" if difference >
                                                0 else "") + f'{difference})'
                mk_down_table += "</td>"

            mk_down_table += f'</td>\n{_tab(3)}</tr>\n'
        mk_down_table += f'{_tab(2)}</tbody>\n{_tab(1)}</table>\n</center>'
        return mk_down_table


if __name__ == "__main__":
    evaluator = Evaluator()
    print(evaluator.print_md_table("MOT20", ["combined"],
                                   ['ByteTrack','CenterTrack'],
                                   columns=[
        "HOTA", "IDSW", "MOTA", "IDF1"], add_baseline=True))
