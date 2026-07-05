TEXT_COLUMN_ALIASES = (
    "Sentence",
    "text",
    "review",
    "review_text",
    "feedback",
    "feedback_text",
    "comment",
    "message",
    "Tweet Text",
)


def find_text_column(columns):
    normalized_columns = {
        str(column).strip().casefold(): column
        for column in columns
    }
    for alias in TEXT_COLUMN_ALIASES:
        matched_column = normalized_columns.get(alias.casefold())
        if matched_column is not None:
            return matched_column
    accepted_columns = ", ".join(TEXT_COLUMN_ALIASES)
    raise ValueError(f"CSV must include one text column: {accepted_columns}")


def normalize_review_dataframe(df):
    text_column = find_text_column(df.columns)
    normalized = df.copy()
    normalized["Sentence"] = normalized[text_column].fillna("").astype(str).str.strip()
    normalized = normalized[normalized["Sentence"] != ""]
    normalized = normalized.drop_duplicates(subset=["Sentence"])
    if normalized.empty:
        raise ValueError("CSV must include at least one non-empty review text row")
    return normalized
