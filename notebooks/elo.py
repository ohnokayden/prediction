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

How to use:
- Feed it a df full of matches which we want to train on-> loop thru 
    until it reaches the latest match
"""

from dataclasses import dataclass, field


BASE_ELO = 1500.0
HOME_ADVANTAGE = 60.0          # added to home rating for win-prob calc only
K_FACTOR = 20.0
SEASON_REGRESSION = 0.7       # fraction pulled toward league mean each season
PROMOTED_PENALTY = 100.0       # promoted teams start this far below league mean


@dataclass
class EloSystem:
    ratings: dict = field(default_factory=dict)
    seen_this_season: set = field(default_factory=set)   # teams active in the CURRENT season
    seen_last_season: set = field(default_factory=set)   # teams active in the season just concluded
    current_season: object = None
    completed_seasons: int = 0   # how many season *transitions* we've been through

    # ---------- core probability model ----------

    def expected_home_win_prob(self, home_elo: float, away_elo: float) -> float:
        """Standard Elo logistic expectation, with home advantage baked in."""
        diff = (home_elo + HOME_ADVANTAGE) - away_elo
        return 1.0 / (1.0 + 10 ** (-diff / 400))

    # ---------- rating access / initialization ----------

    def get_rating(self, team: str) -> float:
        never_seen = team not in self.ratings
        # "Returning" = has a rating on file, but didn't play last season
        # AND hasn't already been re-priced this season (guards against
        # re-triggering the reset on every subsequent match this season).
        returning_after_absence = (
            not never_seen
            and self.completed_seasons > 0
            and team not in self.seen_last_season
            and team not in self.seen_this_season
        )

        if never_seen or returning_after_absence:
            if self.completed_seasons > 0:
                # Either a genuine first-ever promotion, or a team coming
                # back up after being relegated out of the league for a
                # while. Treat both the same way: reset to a fixed
                # below-average rating, since in both cases the squad's
                # current quality is a real unknown relative to a team
                # that's been continuously playing in this league.
                league_mean = sum(self.ratings.values()) / len(self.ratings)
                self.ratings[team] = league_mean - PROMOTED_PENALTY
            else:
                # Still within the very first season we've ever processed:
                # flat baseline for everyone.
                self.ratings[team] = BASE_ELO

        return self.ratings[team]

    # ---------- season boundary handling ----------

    def maybe_start_new_season(self, season) -> None:
        """Call this before processing each match. Detects season changes
        and applies regression-to-mean -- but ONLY to teams that actually
        played last season. Teams sitting out (relegated) are left frozen;
        they get repriced as "returning" the moment they reappear, via
        get_rating(), rather than drifting toward the mean while absent."""
        if self.current_season is None:
            self.current_season = season
            return

        if season != self.current_season:
            # Mean computed only over teams that were actually part of the
            # league last season -- stale relegated teams shouldn't drag
            # on what "average" means for the teams still playing.
            active_ratings = [self.ratings[t] for t in self.seen_this_season]
            league_mean = sum(active_ratings) / len(active_ratings)

            for team in self.seen_this_season:
                self.ratings[team] = (
                    # (1 - SEASON_REGRESSION) * self.ratings[team] + SEASON_REGRESSION * league_mean
                    BASE_ELO + (self.ratings[team] - BASE_ELO) * SEASON_REGRESSION
                )
            # Teams NOT in seen_this_season (i.e. relegated/absent) are
            # simply left untouched here.

            self.seen_last_season = self.seen_this_season
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