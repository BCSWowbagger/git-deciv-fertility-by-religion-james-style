from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

BASE_RELIGION_ORDER = [
	"Buddhist",
	"Catholic",
	"Born-Again Prot",
	"Hindu",
	"Jehovahs Witness",
	"Jewish",
	"Mainline Prot",
	"Mormon",
	"Muslim",
	"Orthodox Christian",
	"Unaffiliated",
]
RELIGION_ORDER = BASE_RELIGION_ORDER

ATTENDANCE_ORDER = [
	"Never",
	"Seldom",
	"A few times a year",
	"Once or twice a month",
	"Once a week",
	"More than once a week",
]

BASE_RELIGION_LABEL_MAP = {
	"Born-Again Prot": "Born-Again",
	"Jehovahs Witness": "J. Witness",
	"Mainline Prot": "Mainline",
	"Orthodox Christian": "Orthodox",
	"Unaffiliated": "None",
}
RELIGION_LABEL_MAP = BASE_RELIGION_LABEL_MAP

BASE_COLOR_MAP = {
	"Mormon": "#3f7fcd",
	"Muslim": "#8e44ad",
	"J. Witness": "#ff9da7",
	"Born-Again": "#e4572e",
	"Mainline": "#76b7b2",
	"Catholic": "#b6d957",
	"Orthodox": "#2ecc71",
	"None": "#f1ce2b",
	"Jewish": "#e69f00",
	"Buddhist": "#7f5f3f",
	"Hindu": "#e84a5f",
}
COLOR_MAP = BASE_COLOR_MAP

DEFAULT_GROUPING_OPTIONS = {
	"indifferentist_mode": "off",
	"indifferentist_label": "Indifferent",
	"indifferentist_attendance_levels": {
		"Never",
		"Seldom",
		"A few times a year",
		"Once or twice a month",
		#"Once a week",
		#"More than once a week"
	},
	"indifferentist_excluded_religions": {
		"Unaffiliated",
		"Hindu",
		"Buddhist",
	},
	"excluded_religions": set(),
}

GROUPING_PRESETS = {
	"default": {
		"indifferentist_mode": "off",
	},
	"indifferentist_aggregate": {
		"indifferentist_mode": "aggregate",
	},
	"indifferentist_split": {
		"indifferentist_mode": "split",
	},
}

DATASET_CONFIGS = {
	"CES_2018": {
		"data_path": BASE_DIR / "CES_2018" / "cces18_common_vv.dta",
		"current_year": 2018,
		"weight_var": "commonweight",
		"gender_column": "gender",
		"default_gender_filter": "Female",
		"default_min_age": 44,
		"default_max_age": 55,
		"default_analysis_mode": "numchildren",
		"value_labels": {
			"numchildren": "completed fertility",
			"child18num": "children under 18",
		},
		"gender_map": {
			None: "adults",
			"Female": "women",
			"Male": "men",
		},
		"data_label": "CES 2018",
		"default_chart_output": BASE_DIR / "religion_fertility_chart_ces2018.png",
	},
	"CES_2020": {
		"data_path": BASE_DIR / "CES2020" / "CES20_Common_OUTPUT_vv.dta",
		"current_year": 2020,
		"weight_var": "commonweight",
		"gender_column": "gender",
		"default_gender_filter": "Female",
		"default_min_age": 44,
		"default_max_age": 55,
		"default_analysis_mode": "numchildren",
		"value_labels": {
			"numchildren": "completed fertility",
			"child18num": "children under 18",
		},
		"gender_map": {
			None: "adults",
			"Female": "women",
			"Male": "men",
		},
		"data_label": "CES 2020",
		"default_chart_output": BASE_DIR / "religion_fertility_chart_ces2020.png",
	},
	"CES_2022": {
		"data_path": BASE_DIR / "CES_2022" / "CCES22_Common_OUTPUT_vv_topost.dta",
		"current_year": 2022,
		"weight_var": "commonweight",
		"gender_column": "gender4",
		"default_gender_filter": "Woman",
		"default_min_age": 44,
		"default_max_age": 55,
		"default_analysis_mode": "numchildren",
		"value_labels": {
			"numchildren": "completed fertility",
			"child18": "share with children under 18 in household",
		},
		"gender_map": {
			None: "adults",
			"Woman": "women",
			"Man": "men",
			"Non-binary": "non-binary adults",
			"Other": "other-gender adults",
		},
		"data_label": "CES 2022",
		"default_chart_output": BASE_DIR / "religion_fertility_chart_ces2022.png",
	},
	"CES_2024": {
		"data_path": BASE_DIR / "CES_2024" / "CCES24_Common_OUTPUT_vv_topost_final.dta",
		"current_year": 2024,
		"weight_var": "commonweight",
		"gender_column": "gender4",
		"default_gender_filter": "Woman",
		"default_min_age": 44,
		"default_max_age": 55,
		"default_analysis_mode": "numchildren",
		"value_labels": {
			"numchildren": "completed fertility",
			"child18": "share with children under 18 in household",
		},
		"gender_map": {
			None: "adults",
			"Woman": "women",
			"Man": "men",
			"Non-binary": "non-binary adults",
			"Other": "other-gender adults",
		},
		"data_label": "CES 2024",
		"default_chart_output": BASE_DIR / "religion_fertility_chart_ces2024.png",
	},
}


def get_dataset_config(dataset_key):
	if dataset_key not in DATASET_CONFIGS:
		raise ValueError(f"Unsupported dataset key: {dataset_key}")
	return DATASET_CONFIGS[dataset_key]


def summarize_group(g, value_col, weight_col):
	values = g[value_col].to_numpy(dtype=float)
	weights = g[weight_col].to_numpy(dtype=float)
	weight_sum = weights.sum()

	if len(g) == 0 or weight_sum <= 0:
		return pd.Series(
			{
				"mean_children": np.nan,
				"n_obs": len(g),
				"weighted_n": weight_sum,
				"n_eff": np.nan,
				"se_mean": np.nan,
				"ci95_low": np.nan,
				"ci95_high": np.nan,
			}
		)

	mean_value = np.average(values, weights=weights)
	n_obs = len(g)
	weighted_n = weight_sum
	sum_w2 = np.square(weights).sum()
	n_eff = (weighted_n ** 2) / sum_w2 if sum_w2 > 0 else np.nan

	centered = values - mean_value
	weighted_var = np.average(np.square(centered), weights=weights)
	se_mean = np.sqrt(weighted_var / n_eff) if n_eff > 0 else np.nan
	ci_low = mean_value - 1.96 * se_mean if pd.notna(se_mean) else np.nan
	ci_high = mean_value + 1.96 * se_mean if pd.notna(se_mean) else np.nan

	return pd.Series(
		{
			"mean_children": mean_value,
			"n_obs": n_obs,
			"weighted_n": weighted_n,
			"n_eff": n_eff,
			"se_mean": se_mean,
			"ci95_low": ci_low,
			"ci95_high": ci_high,
		}
	)


def build_grouping_options(preset="default", overrides=None):
	if preset not in GROUPING_PRESETS:
		raise ValueError(f"Unsupported grouping preset: {preset}")

	options = resolve_grouping_options(GROUPING_PRESETS[preset])
	if overrides:
		options = resolve_grouping_options({**options, **overrides})
	return options


def resolve_grouping_options(grouping_options=None):
	options = {
		"indifferentist_mode": DEFAULT_GROUPING_OPTIONS["indifferentist_mode"],
		"indifferentist_label": DEFAULT_GROUPING_OPTIONS["indifferentist_label"],
		"indifferentist_attendance_levels": set(DEFAULT_GROUPING_OPTIONS["indifferentist_attendance_levels"]),
		"indifferentist_excluded_religions": set(DEFAULT_GROUPING_OPTIONS["indifferentist_excluded_religions"]),
		"excluded_religions": set(DEFAULT_GROUPING_OPTIONS["excluded_religions"]),
	}
	if grouping_options is None:
		return options

	for key, value in grouping_options.items():
		if key in {"indifferentist_attendance_levels", "indifferentist_excluded_religions", "excluded_religions"}:
			options[key] = set(value)
		else:
			options[key] = value

	mode = options["indifferentist_mode"]
	if mode not in {"off", "aggregate", "split"}:
		raise ValueError(f"Unsupported indifferentist_mode: {mode}")
	return options


def get_display_label(religion_name):
	return BASE_RELIGION_LABEL_MAP.get(religion_name, religion_name)


def lighten_hex(hex_color, amount=0.35):
	hex_color = hex_color.lstrip("#")
	r = int(hex_color[0:2], 16)
	g = int(hex_color[2:4], 16)
	b = int(hex_color[4:6], 16)
	r = round(r + (255 - r) * amount)
	g = round(g + (255 - g) * amount)
	b = round(b + (255 - b) * amount)
	return f"#{r:02x}{g:02x}{b:02x}"


def get_religion_display_config(grouping_options=None):
	options = resolve_grouping_options(grouping_options)
	excluded = options["excluded_religions"]
	mode = options["indifferentist_mode"]

	religion_order = []
	label_map = {}
	color_map = {}

	for religion in BASE_RELIGION_ORDER:
		if religion in excluded:
			continue
		religion_order.append(religion)
		label = get_display_label(religion)
		label_map[religion] = label
		color_map[label] = BASE_COLOR_MAP.get(label, "#777777")

		if mode == "split" and religion not in options["indifferentist_excluded_religions"]:
			indiff_name = f"Indifferent {religion}"
			indiff_label = f"Indiff. {label}"
			religion_order.append(indiff_name)
			label_map[indiff_name] = indiff_label
			color_map[indiff_label] = lighten_hex(BASE_COLOR_MAP.get(label, "#777777"), amount=0.45)

	if mode == "aggregate":
		label_map[options["indifferentist_label"]] = options["indifferentist_label"]
		color_map[options["indifferentist_label"]] = "#9a9a9a"
		religion_order.append(options["indifferentist_label"])

	return religion_order, label_map, color_map, options


def build_religion_group(df, grouping_options=None):
	options = resolve_grouping_options(grouping_options)
	df["RELIGION"] = pd.NA

	unaffiliated = {"Nothing in particular", "Agnostic", "Atheist"}
	df.loc[df["religpew"].isin(unaffiliated), "RELIGION"] = "Unaffiliated"

	df.loc[df["religpew"] == "Roman Catholic", "RELIGION"] = "Catholic"
	df.loc[df["religpew"] == "Jewish", "RELIGION"] = "Jewish"
	df.loc[df["religpew"] == "Mormon", "RELIGION"] = "Mormon"
	df.loc[df["religpew"] == "Muslim", "RELIGION"] = "Muslim"
	df.loc[df["religpew"] == "Buddhist", "RELIGION"] = "Buddhist"
	df.loc[df["religpew"] == "Hindu", "RELIGION"] = "Hindu"
	df.loc[df["religpew"] == "Eastern or Greek Orthodox", "RELIGION"] = "Orthodox Christian"

	protestant = df["religpew"] == "Protestant"
	df.loc[protestant & (df["religpew_protestant"] == "Jehovah's Witness"), "RELIGION"] = "Jehovahs Witness"
	df.loc[protestant & (df["pew_bornagain"] == "Yes"), "RELIGION"] = "Born-Again Prot"
	df.loc[
		protestant
		& (df["religpew_protestant"] != "Jehovah's Witness")
		& (df["pew_bornagain"] != "Yes"),
		"RELIGION",
	] = "Mainline Prot"

	df["BASE_RELIGION"] = df["RELIGION"]

	eligible_indifferentists = (
		df["RELIGION"].notna()
		& df["RELIGION"].isin(BASE_RELIGION_ORDER)
		& ~df["RELIGION"].isin(options["indifferentist_excluded_religions"])
		& df["pew_churatd"].isin(options["indifferentist_attendance_levels"])
	)

	if options["indifferentist_mode"] == "aggregate":
		df.loc[eligible_indifferentists, "RELIGION"] = options["indifferentist_label"]
	elif options["indifferentist_mode"] == "split":
		df.loc[eligible_indifferentists, "RELIGION"] = (
			"Indifferent " + df.loc[eligible_indifferentists, "RELIGION"].astype(str)
		)

	return df


def run_analysis(
	dataset_key,
	*,
	analysis_mode=None,
	min_age=None,
	max_age=None,
	gender_filter=None,
	weight_var=None,
	grouping_options=None,
):
	config = get_dataset_config(dataset_key)
	analysis_mode = analysis_mode or config["default_analysis_mode"]
	min_age = config["default_min_age"] if min_age is None else min_age
	max_age = config["default_max_age"] if max_age is None else max_age
	weight_var = weight_var or config["weight_var"]
	#print ("using weight:", weight_var)

	if analysis_mode not in config["value_labels"]:
		raise ValueError(f"Unsupported analysis mode for {dataset_key}: {analysis_mode}")

	columns = [
		"birthyr",
		"religpew",
		"religpew_protestant",
		"pew_bornagain",
		"pew_churatd",
		config["gender_column"],
		weight_var,
	]
	if analysis_mode == "child18":
		columns.append("child18")
	else:
		columns.append(analysis_mode)

	df = pd.read_stata(config["data_path"], columns=columns, convert_categoricals=True)
	df["age"] = config["current_year"] - df["birthyr"]

	if analysis_mode == "child18":
		df["analysis_value"] = np.where(df["child18"] == "Yes", 1.0, np.where(df["child18"] == "No", 0.0, np.nan))
	else:
		df["analysis_value"] = df[analysis_mode]

	df = df[
		df["age"].between(min_age, max_age)
		& df["analysis_value"].notna()
		& df[weight_var].notna()
		& (df[weight_var] > 0)
	].copy()

	if gender_filter is not None:
		df = df[df[config["gender_column"]] == gender_filter].copy()

	religion_order, religion_label_map, religion_color_map, grouping_options = get_religion_display_config(grouping_options)
	df = build_religion_group(df, grouping_options=grouping_options)

	religion_df = df[df["RELIGION"].isin(religion_order)].copy()
	attendance_df = df[df["pew_churatd"].isin(ATTENDANCE_ORDER)].copy()

	religion_pct = (
		religion_df.groupby("RELIGION")[weight_var]
		.sum()
		.div(religion_df[weight_var].sum())
		.mul(100)
		.rename("overall_pct")
	)

	attendance_pct = (
		attendance_df.groupby("pew_churatd")[weight_var]
		.sum()
		.div(attendance_df[weight_var].sum())
		.mul(100)
		.rename("pct_of_attendance_sample")
	)

	religion_summary = (
		religion_df.groupby("RELIGION")
		.apply(summarize_group, value_col="analysis_value", weight_col=weight_var)
		.reindex(religion_order)
		.join(religion_pct)
	)
	religion_summary.loc["Total"] = summarize_group(religion_df, "analysis_value", weight_var)

	attendance_summary = (
		attendance_df.groupby("pew_churatd")
		.apply(summarize_group, value_col="analysis_value", weight_col=weight_var)
		.reindex(ATTENDANCE_ORDER)
		.join(attendance_pct)
	)
	attendance_summary.loc["Total"] = summarize_group(attendance_df, "analysis_value", weight_var)

	return {
		"dataset_key": dataset_key,
		"data_label": config["data_label"],
		"analysis_mode": analysis_mode,
		"analysis_label": config["value_labels"][analysis_mode],
		"current_year": config["current_year"],
		"min_age": min_age,
		"max_age": max_age,
		"gender_filter": gender_filter,
		"gender_column": config["gender_column"],
		"weight_var": weight_var,
		"loaded_n": len(df),
		"religion_n": len(religion_df),
		"attendance_n": len(attendance_df),
		"religion_summary": religion_summary,
		"attendance_summary": attendance_summary,
		"config": config,
		"grouping_options": grouping_options,
		"religion_order": religion_order,
		"religion_label_map": religion_label_map,
		"religion_color_map": religion_color_map,
	}


def get_plot_ready_religion_summary(
	dataset_key,
	*,
	analysis_mode=None,
	min_age=None,
	max_age=None,
	gender_filter=None,
	weight_var=None,
	grouping_options=None,
):
	result = run_analysis(
		dataset_key,
		analysis_mode=analysis_mode,
		min_age=min_age,
		max_age=max_age,
		gender_filter=gender_filter,
		weight_var=weight_var,
		grouping_options=grouping_options,
	)
	return prepare_religion_data_for_plot(result)

def prepare_religion_data_for_plot(religion_data):
	plot_df = (
		religion_data["religion_summary"]
		.drop(index="Total")
		.dropna(subset=["mean_children"])
		.reset_index(names="religion")
		.rename(
			columns={
				"mean_children": "mean",
				"ci95_low": "low",
				"ci95_high": "high",
			}
		)
	)
	plot_df["religion"] = plot_df["religion"].replace(religion_data["religion_label_map"])
	religion_data["plot_religion_summary"] = plot_df
	return religion_data


def choose_national_avg_y(df, national_avg, label_x, preferred_y):
	blocked_rows = df.index[df["high"] >= label_x].to_numpy(dtype=float)
	candidates = np.arange(-0.25, len(df) - 0.75 + 0.001, 0.25)

	best_y = preferred_y
	best_score = None

	for y in candidates:
		if blocked_rows.size == 0:
			clearance = 10.0
		else:
			clearance = np.min(np.abs(blocked_rows - y))
		score = (clearance, -abs(y - preferred_y))
		if best_score is None or score > best_score:
			best_score = score
			best_y = y

	return best_y

def print_analysis_report(result):
	gender_note = result["gender_filter"] if result["gender_filter"] is not None else "All respondents"
	print(
		f"Loaded {result['loaded_n']:,} {result['data_label']} respondents aged {result['min_age']}-{result['max_age']} "
		f"in {result['current_year']} with valid {result['analysis_label']} data."
	)
	print(f"Using weight: {result['weight_var']}")
	print(f"Analysis mode: {result['analysis_mode']} ({result['analysis_label']})")
	print(f"Gender filter: {gender_note}")
	print(f"Grouping options: {result['grouping_options']}")
	print(f"Religion analysis sample: {result['religion_n']:,} respondents after dropping unmapped religions.")
	print(f"Attendance analysis sample: {result['attendance_n']:,} respondents after dropping non-substantive attendance responses.")
	print(f"\nWeighted mean {result['analysis_label']} by religion:")
	print(result["religion_summary"])
	#print(f"\nWeighted mean {result['analysis_label']} by religious attendance:")
	#print(result["attendance_summary"])


def get_and_render_religion_data(
	dataset_key,
	*,
	analysis_mode=None,
	min_age=None,
	max_age=None,
	gender_filter=None,
	weight_var=None,
	grouping_options=None,
	output_path=None,
	show_plot=False,
):
	result = get_plot_ready_religion_summary(
		dataset_key,
		analysis_mode=analysis_mode,
		min_age=min_age,
		max_age=max_age,
		gender_filter=gender_filter,
		weight_var=weight_var,
		grouping_options=grouping_options,
	)
	return render_religion_chart(result, output_path, show_plot)
	

def render_religion_chart(religion_data, output_path=None, show_plot=False):
	religion_data = prepare_religion_data_for_plot(religion_data)
	df = religion_data["plot_religion_summary"].sort_values("mean", ascending=True).copy()
	national_avg = religion_data["religion_summary"].loc["Total", "mean_children"]
	config = religion_data["config"]

	plt.rcParams.update({
		"font.family": "DejaVu Sans",
		"font.size": 12,
	})

	colors = [religion_data["religion_color_map"].get(r, "#777777") for r in df["religion"]]
	fig, ax = plt.subplots(figsize=(8, 6))

	xerr = [df["mean"] - df["low"], df["high"] - df["mean"]]
	ax.barh(
		df["religion"],
		df["mean"],
		xerr=xerr,
		color=colors,
		edgecolor="black",
		linewidth=1.2,
		height=0.9,
		capsize=5,
	)

	for i, value in enumerate(df["mean"]):
		ax.text(0.08, i, f"{value:.2f}", va="center", ha="left", fontsize=12, color="black")

	x_max = max(df["high"]) + 0.8
	n_x = max(df["high"]) + 0.18
	for i, n_obs in enumerate(df["n_obs"]):
		ax.text(
			n_x,
			i,
			f"n={int(n_obs)}",
			va="center",
			ha="left",
			fontsize=10,
			color="#9a9a9a",
			fontstyle="italic",
		)

	ax.axvline(national_avg, linestyle="--", color="black", linewidth=1.5)
	national_label_x = national_avg + 0.07
	#national_label_y = choose_national_avg_y(df, national_avg, national_label_x, len(df) / 2 - 0.5) # this method doesn't work very well yet. Keep guessing around the middle.
	national_label_y = len(df) / 2 - 2
	ax.text(
		national_label_x,
		national_label_y,
		f"National Average:\n{national_avg:.2f}",
		va="center",
		fontsize=13,
	)

	gender_label = config["gender_map"].get(religion_data["gender_filter"], "adults")
	fig.suptitle(
		f"Average Number of Children by Religion\n({religion_data['min_age']} to {religion_data['max_age']} year old {gender_label})",
		fontsize=20,
		weight="bold",
	)
	fig.text(
		0.99,
		0.00,
		f"JamesJHeaney.com | Data: {religion_data['data_label']} | Measure: {religion_data['analysis_label']}",
		ha="right",
		va="bottom",
		fontsize=9,
		color="#444444",
	)

	ax.set_xlabel("")
	ax.set_ylabel("")
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_visible(True)
	ax.spines["left"].set_linewidth(1.2)
	ax.spines["bottom"].set_linewidth(1.2)
	ax.set_axisbelow(True)
	ax.grid(axis="x", linestyle="--", alpha=0.25)
	ax.tick_params(axis="y", length=0)

	for label in ax.get_yticklabels():
		label.set_fontsize(12)
		label.set_fontweight("bold")

	ax.set_xlim(0, x_max)
	plt.tight_layout()
	final_output_path = Path(output_path) if output_path is not None else config["default_chart_output"]
	plt.savefig(final_output_path, dpi=300, bbox_inches="tight")
	if show_plot:
		plt.show()
	else:
		plt.close(fig)

	return final_output_path, religion_data

	
#result = run_analysis(
#		"CES_2022",
		# analysis_mode="numchildren",
		# min_age=44,
		# max_age=55,
		# gender_filter="Female",
		# weight_var="commonweight",
#	)
#print_analysis_report(result)
