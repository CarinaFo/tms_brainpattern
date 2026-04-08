import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
import numpy as np
import re
from matplotlib.backends.backend_pdf import PdfPages

# load csv file from RedCap
df = pd.read_csv(r"C:\Users\CarinaF\Downloads\P3990PatientExperien_DATA_LABELS_2026-04-02_1325.csv")

# ID 225 is ID 224 (Record ID 25), Olivia told me on the 15th of January

# drop no response rows
df = df[(df['Survey Timestamp'] != '[not completed]')]

df = df.dropna(subset=["Patient ID:"])

# dropp empty columns
df = df.dropna(axis=1, how='all')

column_names = df.columns[3:-1]

# --- Save all plots to a single PDF ---
with PdfPages("survey_results.pdf") as pdf:

    # Q1: Age
    fig, ax = plt.subplots(figsize=(6,4))  # smaller, consistent size
    ax = df[column_names[0]].value_counts().sort_index().plot(
        kind="barh",
        figsize=(6, 4),
        color="skyblue"
    )

    # Force x-axis to show integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Labels and title
    plt.xlabel("response count")
    plt.ylabel("")
    plt.title(f'Q1: {column_names[0]}')
    plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # Q2: Gender
    fig, ax = plt.subplots(figsize=(6,4))  # smaller, consistent size
    ax = df[column_names[1]].value_counts().sort_index().plot(
        kind="barh",
        figsize=(6, 4),
        color="skyblue"
    )

    # Force x-axis to show integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Labels and title
    plt.xlabel("response count")
    plt.ylabel("")
    plt.title(f'Q2: {column_names[1]}')
    plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # Q3: Age of depression diagnosis
    fig, ax = plt.subplots(figsize=(6,4))  # smaller, consistent size
    ax = df[column_names[2]].plot(kind="hist", color='skyblue')

    # Force x-axis to show integer ticks
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel(f'Q3: {column_names[2]}')
    plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # Q4: Relapse
    rename_dict = {
        "My current treatment relates to a relapse of depression": "relapse",
        "My current treatment relates to my first diagnosis of depression": "first diagnosis"
    }

    # Replace values in the column
    df[column_names[3]] = df[column_names[3]].replace(rename_dict)

    fig, ax = plt.subplots(figsize=(6,4))  # smaller, consistent size
    ax = df[column_names[3]].value_counts().sort_index().plot(
        kind="barh",
        figsize=(6, 4),
        color="skyblue"
    )

    # Force x-axis to show integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Labels and title
    plt.xlabel("response count")
    plt.ylabel("")
    plt.title(f'Q4: {column_names[3]}')
    plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # Q5: How long ago diagnosed?
    df[column_names[4]] = df[column_names[4]].apply(convert_age)

    fig, ax = plt.subplots(figsize=(6,4))  # smaller, consistent size
    ax = df[column_names[4]].plot(kind="hist", color='skyblue')
    plt.title('Q5: How long ago have you been diagnosed with MDD?')
    plt.xlabel('Years since diagnosis')
    plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # Q6: Therapy and experience with it
    fig, ax = plt.subplots(figsize=(6,4))  # smaller, consistent size
    ax = df[column_names[5]].value_counts().sort_index().plot(
        kind="barh",
        figsize=(6, 4),
        color="skyblue"
    )

    # Force x-axis to show integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Labels and title
    plt.xlabel("response count")
    plt.ylabel("")
    plt.title(f'Q6: {column_names[5]}')
    plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # Q7: Other treatments
    fig, ax = plt.subplots(figsize=(6,4))  # smaller, consistent size
    ax = df[column_names[6]].value_counts().sort_index().plot(
        kind="barh",
        figsize=(6, 4),
        color="skyblue"
    )

    # Force x-axis to show integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Labels and title
    plt.xlabel("response count")
    plt.ylabel("")
    plt.title(f'Q7: {column_names[6]}')
    plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # Q8: How did you learn about TMS treatment?
    fig, ax = plt.subplots(figsize=(6,4))  # smaller, consistent size
    ax = df[column_names[7]].value_counts().sort_index().plot(
        kind="barh",
        figsize=(6, 4),
        color="skyblue"
    )

    # Force x-axis to show integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Labels and title
    plt.xlabel("response count")
    plt.ylabel("")
    plt.title(f'Q8: {column_names[7]}')
    plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # Q9: Support during treatment
    support_cols = df.filter(like="Who supports you")
    df_bool = support_cols.applymap(lambda x: x == 'Checked')

    # clean column names: keep everything after "choice="
    df_bool.columns = (
        df_bool.columns
        .str.extract(r'choice=(.*)\)$')[0]   # extract text after choice=
        .str.strip()
    )

    support_summary = df_bool.sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6,4))  # smaller, consistent size
    ax = support_summary.plot(kind='barh', color='skyblue')
    plt.xlabel("response count")
    # Force x-axis to show integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Q9: Who supports you in your treatment journey?")
    plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # Q 10: health professionals visits
    visit_cols = df.columns[df.apply(lambda col: col.astype(str).str.contains(">4 Times").any())]

    # count values per response
    summary = {
        col: df[col].value_counts(dropna=True)
        for col in visit_cols
    }

    # convert to dataframe
    summary_df = pd.DataFrame(summary).fillna(0).astype(int)

    order = [
        "Never",
        "Once",
        "Twice",
        "3 Times",
        "4 Times",
        ">4 Times"
    ]

    # order the index
    summary_df = summary_df.reindex(order)

    plot_df = summary_df.reset_index().melt(
        id_vars="index",
        var_name="Question",
        value_name="Count"
    ).rename(columns={"index": "Response"})


    fig, ax = plt.subplots(figsize=(10,6))
    ax = sns.barplot(
        data=plot_df,
        y="Response",
        x="Count",
        hue="Question"
    )

    plt.title("Q10: In the 12 months prior to starting TMS, how many times did you visit ... ?")
    plt.xlabel("response count")
    plt.ylabel("")
    plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # Q11: Treatment understanding
    fig, ax = plt.subplots(figsize=(6,4))  # smaller, consistent size
    ax = df[column_names[22]].value_counts().sort_index().plot(
        kind="barh",
        figsize=(6, 4),
        color="skyblue"
    )

    # Force x-axis to show integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Labels and title
    plt.xlabel("response count")
    plt.ylabel("")
    plt.title(f'Q11: {column_names[22]}')
    plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # Q 12: treatment expectation
    support_cols = df.filter(like="Which of these")
    df_bool = support_cols.applymap(lambda x: x == 'Checked')

    # clean column names: keep everything after "choice="
    df_bool.columns = (
        df_bool.columns
        .str.extract(r'choice=(.*)\)$')[0]   # extract text after choice=
        .str.strip()
    )

    support_summary = df_bool.sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6,4))  # smaller, consistent size
    ax = support_summary.plot(kind='barh', color='skyblue')
    plt.xlabel("response count")
    plt.title("Q12: Which of these match your expectations of TMS?")
    plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # Q13: Factors of Importance for TMS treatment
    # select columns
    importance_cols = ['Cost', "Treatment success rate", "Side effects (i.e. headache)", "Treatment duration", "Doctor recommendation",
                        "Recovery time", "Convenience of treatment"]

    # count values per response
    summary = {
        col: df[col].value_counts(dropna=True)
        for col in importance_cols
    }

    # convert to dataframe
    summary_df = pd.DataFrame(summary).fillna(0).astype(int)

    order = [
        "1 (Least)",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7 (most)"
    ]

    # order the index
    summary_df = summary_df.reindex(order)

    plot_df = summary_df.reset_index().melt(
        id_vars="index",
        var_name="Question",
        value_name="Count"
    ).rename(columns={"index": "Response"})


    fig, ax = plt.subplots(figsize=(10,6))
    ax = sns.barplot(
        data=plot_df,
        y="Response",
        x="Count",
        hue="Question"
    )

    plt.title("Q13: Importance of factors when considering TMS treatment ?")
    plt.xlabel("response count")
    plt.ylabel("Importance")
    plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # Q14: difficulty of getting treatment
    fig, ax = plt.subplots(figsize=(6,4))  # smaller, consistent size
    ax = df["How easy or difficult has it been for you to access TMS?"].value_counts().plot(kind="barh", color='skyblue')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('response count')
    plt.ylabel("")
    plt.title("Q14: How easy or difficult has it been for you to access TMS?")
    plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


    # Q15: Barriers for treatment
    support_cols = df.filter(like="Did you experience")
    df_bool = support_cols.applymap(lambda x: x == 'Checked')

    # clean column names: keep everything after "choice="
    df_bool.columns = (
        df_bool.columns
        .str.extract(r'choice=(.*)\)$')[0]   # extract text after choice=
        .str.strip()
    )

    support_summary = df_bool.sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6,4))  # smaller, consistent size
    ax = support_summary.plot(kind='barh', color='skyblue')
    plt.xlabel("response count")
    plt.title("Q15: Did you experience any barriers in TMS treatment?")
    plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # Q16: Fears or concerns
    fig, ax = plt.subplots(figsize=(6,4))  # smaller, consistent size
    ax = df["Do you have any concerns or fears about the TMS treatment before starting?"].value_counts().plot(kind="barh", color='skyblue')
    plt.xlabel('response count')
    plt.ylabel("")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Q16: Do you have any concerns or fears about the TMS treatment before starting?")
    plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # Q17: Test predicts treatment outcome
    # select columns
    test_cols = df.filter(like="This information")

    # count values per response
    summary = {
        col: df[col].value_counts(dropna=True)
        for col in test_cols
    }

    # convert to dataframe
    summary_df = pd.DataFrame(summary).fillna(0).astype(int)

    summary_df = summary_df.rename(columns={
        summary_df.columns[0]: "Very valuable",
        "This information would be valuable to me, and it would be good to know at the point in my treatment pathway that I was considering TMS therapy": "Valuable",
        "This information would have no real value to me": "Not valuable",
        "This information would cause me worry and anxiety and I would prefer to not have this information": "would cause me anxiety and worry"
    })

    order = [
        "Not true at all",
        "Not very true",
        "Somewhat true",
        "Very true"
    ]

    # order the index
    summary_df = summary_df.reindex(order)

    plot_df = summary_df.reset_index().melt(
        id_vars="index",
        var_name="Question",
        value_name="Count"
    ).rename(columns={"index": "Response"})


    fig, ax = plt.subplots(figsize=(10,6))
    ax = sns.barplot(
        data=plot_df,
        y="Response",
        x="Count",
        hue="Question"
    )

    plt.title("Q17: Usefulness of TMS treatment success test?")
    plt.xlabel("response count")
    plt.ylabel("")
    plt.tight_layout()
    plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # Q18: Test result from EEG scan
    test_cols = df.filter(like="The result")

    # count values per response
    summary = {
        col: df[col].value_counts(dropna=True)
        for col in test_cols
    }

    # convert to dataframe
    summary_df = pd.DataFrame(summary).fillna(0).astype(int)

    summary_df = summary_df.rename(columns={
        summary_df.columns[0]: "10 % chance of response",
        summary_df.columns[1]: "30 % chance of response",
        summary_df.columns[2]: "50 % chance of response",
        summary_df.columns[3]: "70 % chance of response",
        summary_df.columns[4]: "90 % chance of response"
    })

    order = [
        "Probably proceed",
        "Probably not proceed",
        "I'm not sure"
    ]

    # order the index
    summary_df = summary_df.reindex(order)

    plot_df = summary_df.reset_index().melt(
        id_vars="index",
        var_name="Question",
        value_name="Count"
    ).rename(columns={"index": "Response"})


    fig, ax = plt.subplots(figsize=(10,6))
    ax = sns.barplot(
        data=plot_df,
        y="Response",
        x="Count",
        hue="Question"
    )

    plt.title("Q18: Response to TMS treatment success test results")
    plt.xlabel("response count")
    plt.ylabel("")
    plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # Q19: Continue treatment based on AI prediction
    test_cols = df.filter(like="Based on these")

    # count values per response
    summary = {
        col: df[col].value_counts(dropna=True)
        for col in test_cols
    }

    # convert to dataframe
    summary_df = pd.DataFrame(summary).fillna(0).astype(int)

    summary_df = summary_df.rename(columns={
        summary_df.columns[0]: "10 % chance of response",
        summary_df.columns[1]: "30 % chance of response",
        summary_df.columns[2]: "50 % chance of response",
        summary_df.columns[3]: "70 % chance of response",
        summary_df.columns[4]: "90 % chance of response"
    })

    order = [
        "Probably continue treatment",
        "Probably discontinue treatment",
        "I'm not sure"
    ]

    # order the index
    summary_df = summary_df.reindex(order)

    plot_df = summary_df.reset_index().melt(
        id_vars="index",
        var_name="Question",
        value_name="Count"
    ).rename(columns={"index": "Response"})


    fig, ax = plt.subplots(figsize=(10,6))
    ax = sns.barplot(
        data=plot_df,
        y="Response",
        x="Count",
        hue="Question"
    )

    plt.title("Q19: AI treatment success prediction after 10 TMS sesssions without improvement")
    plt.ylabel("")
    plt.xlabel("response count")
    plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# Helper functions
def convert_age(entry):
    if pd.isna(entry) or str(entry).strip() == "":
        return np.nan
    
    entry = str(entry).lower().strip()

    # extract number
    num = re.findall(r"\d+\.?\d*", entry)
    if not num:
        return np.nan
    value = float(num[0])

    # classify unit
    if "month" in entry or "mo" in entry or "m " in entry:
        return value / 12   # convert months → years
    
    return value  # assume years if no unit or "yr", "y", etc.

