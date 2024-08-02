import blobfile as bf
import pandas as pd

languages = [
    "AR-XY",
    "BN-BD",
    "DE-DE",
    "ES-LA",
    "FR-FR",
    "HI-IN",
    "ID-ID",
    "IT-IT",
    "JA-JP",
    "KO-KR",
    "PT-BR",
    "ZH-CN",
]

# Define the header for the CSV files
csv_header = ["Question", "A", "B", "C", "D", "Answer"]


def load_files_from_directory(directory_path):
    data_frames_by_language = {lang: [] for lang in languages}
    for filename in bf.listdir(directory_path):
        if filename.endswith(".csv"):
            for lang in languages:
                if lang in filename:
                    file_path = bf.join(directory_path, filename)
                    with bf.BlobFile(file_path, "r") as f:
                        df = pd.read_csv(f, header=None, names=csv_header)
                        df["filename"] = filename  # Add the filename to the dataframe
                        df[""] = (
                            df.index
                        )  # Add the original index as a column with an empty string as the name
                        data_frames_by_language[lang].append(df)
                    break
    return data_frames_by_language


directory_path = "az://oaidatasets2/evallib/mmlu_human_translated_fixed/test"
data_frames_by_language = load_files_from_directory(directory_path)

for lang in data_frames_by_language:
    if data_frames_by_language[lang]:
        # Concatenate all dataframes for the current language
        concatenated_df = pd.concat(data_frames_by_language[lang], ignore_index=True)

        # Add a final column which consists of filename.split("_test_")[0]
        concatenated_df["Subject"] = concatenated_df["filename"].apply(
            lambda x: x.split("_test_")[0]
        )

        # Replace the list of dataframes with the concatenated dataframe
        data_frames_by_language[lang] = [concatenated_df]

for lang, dfs in data_frames_by_language.items():
    if dfs:
        # Drop the 'filename' column before exporting
        dfs[0] = dfs[0].drop(columns=["filename"])

        # Reorder columns to put the empty string column first
        columns = [""] + [col for col in dfs[0].columns if col != ""]
        dfs[0] = dfs[0][columns]

        print(f"Example for language {lang}:")
        print(dfs[0].head())  # Print the first few rows of the concatenated dataframe

    output_file_path = f"mmlu_{lang}.csv"
    dfs[0].to_csv(
        output_file_path, index=False
    )  # Ensure the index is not exported as a separate column
    print(f"Saved concatenated dataframe to {output_file_path}")
