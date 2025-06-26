import pandas as pd
from collections import Counter

class SearchAnalytics:
    def __init__(self, log_file="search_log.csv"):
        self.log_file = log_file
        self.log_data = pd.DataFrame(columns=["query", "results_clicked"])
        self.load_log()

    def load_log(self):
        try:
            self.log_data = pd.read_csv(self.log_file)
        except FileNotFoundError:
            pass

    def save_log(self):
        self.log_data.to_csv(self.log_file, index=False)

    def log_search(self, query, results_clicked=None):
        new_entry = {"query": query, "results_clicked": results_clicked}
        self.log_data = pd.concat([self.log_data, pd.DataFrame([new_entry])], ignore_index=True)
        self.save_log()

    def get_top_searches(self, top_n=10):
        query_counts = Counter(self.log_data["query"])
        return query_counts.most_common(top_n)

    def get_click_through_rate(self):
        if len(self.log_data) == 0:
            return 0
        clicked_searches = self.log_data["results_clicked"].dropna()
        if len(clicked_searches) == 0:
            return 0
        return len(clicked_searches) / len(self.log_data)

    def get_search_counts(self):
        return len(self.log_data)