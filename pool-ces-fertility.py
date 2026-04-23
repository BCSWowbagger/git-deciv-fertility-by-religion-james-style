import numpy as np
import pandas as pd

from ces_fertility_core import (
	ATTENDANCE_ORDER,
	build_grouping_options,
	build_religion_group,
	get_dataset_config,
	get_religion_display_config,
	print_analysis_report,
	render_religion_chart,
	summarize_group,
)


WAVES = ["CES_2018", "CES_2020", "CES_2022", "CES_2024"]
ANALYSIS_MODE = "numchildren"

MIN_AGE = 44
MAX_AGE = 55

GENDER_PRESET = "Female" # "Female" or None, could later make support for men.

if GENDER_PRESET == "Female":
	GENDER_FILTERS = {
		"CES_2018": "Female",
		"CES_2020": "Female",
		"CES_2022": "Woman",
		"CES_2024": "Woman",
	}
else:
	GENDER_FILTERS = {
		"CES_2018": None,
		"CES_2020": None,
		"CES_2022": None,
		"CES_2024": None,
	}

POOLING_METHOD = "stacked_within_wave_normalized"
SUPPRESS_SINGLE_OBSERVATION_POOLS = False

GROUPING_PRESET = "indifferentist_split" #"indifferentist_aggregate", "indifferentist_split", "default" (aka none)
GROUPING_OVERRIDES = {
	"indifferentist_label": "Indifferent",
	"excluded_religions": set(),
}

SAVE_OUTPUTS = True
OUTPUT_PREFIX = "pooled_ces_fertility"
RENDER_CHART = True
CHART_OUTPUT_PATH = "religion_fertility_chart_pooled.png"
SHOW_PLOT = True


def load_wave(dataset_key, analysis_mode, min_age, max_age, gender_filter, grouping_options):
	config = get_dataset_config(dataset_key)

	if analysis_mode not in config["value_labels"]:
		raise ValueError(f"{dataset_key} does not support analysis mode {analysis_mode}")

	columns = [
		"birthyr",
		"religpew",
		"religpew_protestant",
		"pew_bornagain",
		"pew_churatd",
		config["gender_column"],
		config["weight_var"],
		analysis_mode,
	]

	df = pd.read_stata(config["data_path"], columns=columns, convert_categoricals=True)
	df["age"] = config["current_year"] - df["birthyr"]
	df["analysis_value"] = df[analysis_mode]

	df = df[
		df["age"].between(min_age, max_age)
		& df["analysis_value"].notna()
		& df[config["weight_var"]].notna()
		& (df[config["weight_var"]] > 0)
	].copy()

	if gender_filter is not None:
		df = df[df[config["gender_column"]] == gender_filter].copy()

	df = build_religion_group(df, grouping_options)
	df["wave"] = dataset_key
	df["wave_year"] = config["current_year"]
	df["gender_std"] = gender_filter if gender_filter is not None else "All respondents"
	df["weight_raw"] = df[config["weight_var"]]
	df["weight"] = df["weight_raw"]
	total_weight = df["weight_raw"].sum()
	df["weight_normalized"] = df["weight_raw"] / total_weight if total_weight > 0 else df["weight_raw"]

	return df[
		[
			"wave",
			"wave_year",
			"age",
			"analysis_value",
			"RELIGION",
			"BASE_RELIGION",
			"pew_churatd",
			"weight_raw",
			"weight",
			"weight_normalized",
			"gender_std",
		]
	].copy()


def append_wave_totals(summary_df, stacked_df, group_col, valid_order):
	totals = []
	for wave, wave_df in stacked_df.groupby("wave"):
		if group_col == "RELIGION":
			wave_subset = wave_df[wave_df["RELIGION"].isin(valid_order)].copy()
		else:
			wave_subset = wave_df[wave_df["pew_churatd"].isin(ATTENDANCE_ORDER)].copy()
		total_row = summarize_group(wave_subset, "analysis_value", "weight").to_dict()
		total_row[group_col] = "Total"
		total_row["wave"] = wave
		total_row["pct_of_wave_sample"] = np.nan
		totals.append(total_row)
	if totals:
		summary_df = pd.concat([summary_df, pd.DataFrame(totals)], ignore_index=True)
	return summary_df


def summarize_by_wave(stacked_df, religion_order):
	religion_df = stacked_df[stacked_df["RELIGION"].isin(religion_order)].copy()
	attendance_df = stacked_df[stacked_df["pew_churatd"].isin(ATTENDANCE_ORDER)].copy()

	by_wave_religion = (
		religion_df.groupby(["wave", "RELIGION"])
		.apply(summarize_group, value_col="analysis_value", weight_col="weight")
		.reset_index()
	)
	religion_pct = (
		religion_df.groupby(["wave", "RELIGION"])["weight"]
		.sum()
		.groupby(level=0)
		.transform(lambda s: s / s.sum() * 100 if s.sum() > 0 else np.nan)
		.rename("pct_of_wave_sample")
		.reset_index(drop=True)
	)
	by_wave_religion["pct_of_wave_sample"] = religion_pct
	by_wave_religion = append_wave_totals(by_wave_religion, stacked_df, "RELIGION", religion_order)

	by_wave_attendance = (
		attendance_df.groupby(["wave", "pew_churatd"])
		.apply(summarize_group, value_col="analysis_value", weight_col="weight")
		.reset_index()
	)
	attendance_pct = (
		attendance_df.groupby(["wave", "pew_churatd"])["weight"]
		.sum()
		.groupby(level=0)
		.transform(lambda s: s / s.sum() * 100 if s.sum() > 0 else np.nan)
		.rename("pct_of_wave_sample")
		.reset_index(drop=True)
	)
	by_wave_attendance["pct_of_wave_sample"] = attendance_pct
	by_wave_attendance = append_wave_totals(by_wave_attendance, stacked_df, "pew_churatd", ATTENDANCE_ORDER)

	return by_wave_religion, by_wave_attendance


def summarize_stacked(stacked_df, religion_order):
	religion_df = stacked_df[stacked_df["RELIGION"].isin(religion_order)].copy()
	attendance_df = stacked_df[stacked_df["pew_churatd"].isin(ATTENDANCE_ORDER)].copy()

	pooled_religion = (
		religion_df.groupby("RELIGION")
		.apply(summarize_group, value_col="analysis_value", weight_col="weight_normalized")
		.reindex(religion_order)
	)
	religion_pct = (
		religion_df.groupby("RELIGION")["weight_normalized"]
		.sum()
		.div(religion_df["weight_normalized"].sum())
		.mul(100)
		.rename("pct_of_sample")
	)
	pooled_religion = pooled_religion.join(religion_pct)
	pooled_religion.loc["Total"] = summarize_group(religion_df, "analysis_value", "weight_normalized")
	#pooled_religion["pooling_method"] = "stacked_within_wave_normalized"
	pooled_religion["n_waves"] = len(WAVES)
	#pooled_religion["suppressed_single_obs"] = False

	pooled_attendance = (
		attendance_df.groupby("pew_churatd")
		.apply(summarize_group, value_col="analysis_value", weight_col="weight_normalized")
		.reindex(ATTENDANCE_ORDER)
	)
	attendance_pct = (
		attendance_df.groupby("pew_churatd")["weight_normalized"]
		.sum()
		.div(attendance_df["weight_normalized"].sum())
		.mul(100)
		.rename("pct_of_sample")
	)
	pooled_attendance = pooled_attendance.join(attendance_pct)
	pooled_attendance.loc["Total"] = summarize_group(attendance_df, "analysis_value", "weight_normalized")
	pooled_attendance["pooling_method"] = "stacked_within_wave_normalized"
	pooled_attendance["n_waves"] = len(WAVES)
	pooled_attendance["suppressed_single_obs"] = False

	return pooled_religion, pooled_attendance


def inverse_variance_pool(by_wave_summary, group_col, order):
	pooled_rows = []
	for group_name in order + ["Total"]:
		g = by_wave_summary[by_wave_summary[group_col] == group_name].copy()
		g = g[g["wave"] != "Total"].copy()
		if SUPPRESS_SINGLE_OBSERVATION_POOLS:
			g = g[g["n_obs"] > 1].copy()
		g = g[g["se_mean"].notna() & (g["se_mean"] > 0)].copy()
		if g.empty:
			continue

		inv_var = 1.0 / np.square(g["se_mean"].to_numpy(dtype=float))
		mean_values = g["mean_children"].to_numpy(dtype=float)

		pooled_mean = np.sum(inv_var * mean_values) / np.sum(inv_var)
		pooled_se = np.sqrt(1.0 / np.sum(inv_var))
		pooled_low = pooled_mean - 1.96 * pooled_se
		pooled_high = pooled_mean + 1.96 * pooled_se

		pooled_rows.append(
			{
				group_col: group_name,
				"mean_children": pooled_mean,
				"n_obs": g["n_obs"].sum(),
				#"weighted_n": g["weighted_n"].sum(),
				"n_eff": np.nan,
				"se_mean": pooled_se,
				"ci95_low": pooled_low,
				"ci95_high": pooled_high,
				"n_waves": len(g),
				#"pooling_method": POOLING_METHOD,
				#"suppressed_single_obs": SUPPRESS_SINGLE_OBSERVATION_POOLS,
			}
		)

	pooled_df = pd.DataFrame(pooled_rows)
	if pooled_df.empty:
		return pooled_df

	return pooled_df.set_index(group_col).reindex(order + ["Total"])


def wave_diagnostics(by_wave_religion, religion_order):
	by_wave_religion = by_wave_religion[by_wave_religion["RELIGION"] != "Total"].copy()
	diagnostics = (
		by_wave_religion.groupby("RELIGION")["mean_children"]
		.agg(["count", "min", "max", "mean", "std"])
		.rename(columns={"count": "n_waves", "mean": "mean_of_wave_means", "std": "sd_across_waves"})
	)
	diagnostics["range_across_waves"] = diagnostics["max"] - diagnostics["min"]
	return diagnostics.reindex(religion_order)


def maybe_save_outputs(pooled_religion, pooled_attendance, by_wave_religion, by_wave_attendance, diagnostics):
	if not SAVE_OUTPUTS:
		return

	pooled_religion.to_csv(f"{OUTPUT_PREFIX}_pooled_religion.csv")
	pooled_attendance.to_csv(f"{OUTPUT_PREFIX}_pooled_attendance.csv")
	by_wave_religion.to_csv(f"{OUTPUT_PREFIX}_by_wave_religion.csv", index=False)
	by_wave_attendance.to_csv(f"{OUTPUT_PREFIX}_by_wave_attendance.csv", index=False)
	diagnostics.to_csv(f"{OUTPUT_PREFIX}_wave_diagnostics.csv")


def build_pooled_chart_payload(pooled_religion, religion_order, religion_label_map, religion_color_map, grouping_options):
	return {
		"religion_summary": pooled_religion,
		"config": {
			"gender_map": {
				None: "adults",
				"Female": "women",
				"Male": "men",
				"Woman": "women",
				"Man": "men",
				"Non-binary": "non-binary adults",
				"Other": "other-gender adults",
			},
			"default_chart_output": CHART_OUTPUT_PATH,
		},
		"gender_filter": GENDER_FILTERS.get("CES_2024"),
		"min_age": MIN_AGE,
		"max_age": MAX_AGE,
		"data_label": "Pooled CES " + ", ".join(wave.split("_")[-1] for wave in WAVES),
		"analysis_label": "completed fertility",
		"religion_order": religion_order,
		"religion_label_map": religion_label_map,
		"religion_color_map": religion_color_map,
		"grouping_options": grouping_options,
	}


def main():
	if ANALYSIS_MODE != "numchildren":
		raise ValueError("This pooled script currently supports only ANALYSIS_MODE = 'numchildren'.")

	grouping_options = build_grouping_options(GROUPING_PRESET, GROUPING_OVERRIDES)
	religion_order, religion_label_map, religion_color_map, grouping_options = get_religion_display_config(grouping_options)

	wave_frames = []
	for wave in WAVES:
		gender_filter = GENDER_FILTERS.get(wave)
		wave_frames.append(
			load_wave(
				dataset_key=wave,
				analysis_mode=ANALYSIS_MODE,
				min_age=MIN_AGE,
				max_age=MAX_AGE,
				gender_filter=gender_filter,
				grouping_options=grouping_options,
			)
		)

	stacked_df = pd.concat(wave_frames, ignore_index=True)

	by_wave_religion, by_wave_attendance = summarize_by_wave(stacked_df, religion_order)
	if POOLING_METHOD == "inverse_variance":
		pooled_religion = inverse_variance_pool(by_wave_religion, "RELIGION", religion_order)
		pooled_attendance = inverse_variance_pool(by_wave_attendance, "pew_churatd", ATTENDANCE_ORDER)
	elif POOLING_METHOD == "stacked_within_wave_normalized":
		pooled_religion, pooled_attendance = summarize_stacked(stacked_df, religion_order)
	else:
		raise ValueError(f"Unsupported POOLING_METHOD: {POOLING_METHOD}")
	diagnostics = wave_diagnostics(by_wave_religion, religion_order)

	print("Pooled CES fertility analysis")
	print(f"Waves: {', '.join(WAVES)}")
	print(f"Analysis mode: {ANALYSIS_MODE}")
	print(f"Age range: {MIN_AGE}-{MAX_AGE}")
	print(f"Grouping preset: {GROUPING_PRESET}")
	print(f"Grouping options: {grouping_options}")
	print(f"Pooling method: {POOLING_METHOD}")
	print(f"Suppress single-observation pools: {SUPPRESS_SINGLE_OBSERVATION_POOLS}")
	print(f"Pooled religion analysis sample: {int(by_wave_religion.loc[by_wave_religion['RELIGION'] == 'Total', 'n_obs'].sum()):,}")
	print(f"Pooled attendance analysis sample: {int(by_wave_attendance.loc[by_wave_attendance['pew_churatd'] == 'Total', 'n_obs'].sum()):,}")

	print("\nPer-wave usable samples:")
	for wave in WAVES:
		wave_n = (stacked_df["wave"] == wave).sum()
		print(f"{wave}: {wave_n:,} usable cases")

	print("\nPooled weighted mean completed fertility by religion:")
	print(pooled_religion)

	print("\nPer-wave weighted mean completed fertility by religion:")
	print(by_wave_religion.sort_values(["wave", "RELIGION"]))

	print("\nWave-to-wave diagnostics for religion means:")
	print(diagnostics)

	maybe_save_outputs(
		pooled_religion=pooled_religion,
		pooled_attendance=pooled_attendance,
		by_wave_religion=by_wave_religion,
		by_wave_attendance=by_wave_attendance,
		diagnostics=diagnostics,
	)

	if RENDER_CHART:
		pooled_chart_data = build_pooled_chart_payload(
			pooled_religion,
			religion_order,
			religion_label_map,
			religion_color_map,
			grouping_options,
		)
		render_religion_chart(pooled_chart_data, CHART_OUTPUT_PATH, SHOW_PLOT)


if __name__ == "__main__":
	main()
