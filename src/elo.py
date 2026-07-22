"""
Elo rating system for football match prediction.

Design decisions (locked in):
- All teams start at a flat BASE_ELO (1500) the first time they're ever seen.
- Home advantage is modeled as a constant added to the home team's rating
  ONLY for the purpose of computing win probability (not stored in the
  team's persistent rating).
- K-factor controls how much a single result moves a rating.
- At each season boundary, every team's rating regresses toward the
  league mean (squads change over summer, so full carry-over is wrong).
- A team appearing for the first time after being outside the league
  (i.e. promoted, or first ever season) starts at a fixed BELOW-average
  rating, since promoted teams are usually relegation-battle candidates.
- Every function here only uses information available strictly BEFORE
  the match being processed -- ratings are updated AFTER computing the
  probability/prediction for a match, never before.
"""

from dataclasses import dataclass, field


BASE_ELO = 1500.0
HOME_ADVANTAGE = 60.0          # added to home rating for win-prob calc only
K_FACTOR = 20.0
SEASON_REGRESSION = 0.25       # fraction pulled toward league mean each season
PROMOTED_PENALTY = 100.0       # promoted teams start this far below league mean


@dataclass
class EloSystem:
    ratings: dict = field(default_factory=dict)
    seen_this_season: set = field(default_factory=set)
    current_season: object = None
    completed_seasons: int = 0   # how many season *transitions* we've been through

    # ---------- core probability model ----------

    def expected_home_win_prob(self, home_elo: float, away_elo: float) -> float:
        """Standard Elo logistic expectation, with home advantage baked in."""
        diff = (home_elo + HOME_ADVANTAGE) - away_elo
        return 1.0 / (1.0 + 10 ** (-diff / 400))

    # ---------- rating access / initialization ----------

    def get_rating(self, team: str) -> float:
        if team not in self.ratings:
            # First time we've ever seen this team.
            if self.completed_seasons > 0:
                # We've already been through at least one season boundary,
                # so a brand-new team now is a genuine promotion, not just
                # a team we haven't reached yet in the current season's
                # chronological match list.
                league_mean = sum(self.ratings.values()) / len(self.ratings)
                self.ratings[team] = league_mean - PROMOTED_PENALTY
            else:
                # Still within the very first season we've ever processed:
                # every team is "new" simply because we haven't hit their
                # first fixture yet. Flat baseline for all of them.
                self.ratings[team] = BASE_ELO
        return self.ratings[team]

    # ---------- season boundary handling ----------

    def maybe_start_new_season(self, season) -> None:
        """Call this before processing each match. Detects season changes
        and applies regression-to-mean for every team seen so far."""
        if self.current_season is None:
            self.current_season = season
            return

        if season != self.current_season:
            league_mean = sum(self.ratings.values()) / len(self.ratings)
            for team in self.ratings:
                self.ratings[team] = (
                    (1 - SEASON_REGRESSION) * self.ratings[team]
                    + SEASON_REGRESSION * league_mean
                )
            self.current_season = season
            self.seen_this_season = set()
            self.completed_seasons += 1

    # ---------- match processing ----------

    def process_match(self, home: str, away: str, result: str, season) -> dict:
        """
        result: 'H', 'D', or 'A' (from football-data.co.uk FTR column)

        Returns the pre-match ratings and predicted probability so you can
        use them as FEATURES for that match (this is the leakage-safe part:
        we read ratings, compute probability, THEN update).
        """
        self.maybe_start_new_season(season)

        home_elo_before = self.get_rating(home)
        away_elo_before = self.get_rating(away)

        prob_home_win = self.expected_home_win_prob(home_elo_before, away_elo_before)

        # Convert result to actual score for Elo update purposes.
        # Draw counts as 0.5 for both sides, standard chess-Elo convention.
        actual_home_score = {"H": 1.0, "D": 0.5, "A": 0.0}[result]

        new_home_elo = home_elo_before + K_FACTOR * (actual_home_score - prob_home_win)
        new_away_elo = away_elo_before + K_FACTOR * ((1 - actual_home_score) - (1 - prob_home_win))

        self.ratings[home] = new_home_elo
        self.ratings[away] = new_away_elo

        self.seen_this_season.add(home)
        self.seen_this_season.add(away)

        return {
            "home_elo_pre_match": home_elo_before,
            "away_elo_pre_match": away_elo_before,
            "elo_prob_home_win": prob_home_win,
        }