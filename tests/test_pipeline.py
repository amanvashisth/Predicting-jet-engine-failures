import unittest

import numpy as np
import pandas as pd

from pipeline import RAW_COLUMNS, SENSOR_COLUMNS, engineer_features


class EngineerFeaturesTests(unittest.TestCase):
    def test_engineers_rolling_statistics_independently_per_engine(self):
        records = []
        for unit, offset in ((1, 0), (2, 100)):
            for cycle in range(1, 16):
                row = {column: 0.0 for column in RAW_COLUMNS}
                row.update(
                    {
                        "unit": unit,
                        "cycle": cycle,
                        "set1": 0.1,
                        "set2": 0.2,
                        "set3": 100,
                    }
                )
                for sensor in SENSOR_COLUMNS:
                    row[sensor] = offset + cycle
                records.append(row)

        engineered = engineer_features(pd.DataFrame(records))
        final_cycle = engineered.loc[engineered["cycle"].eq(15)].set_index("unit")

        self.assertAlmostEqual(final_cycle.loc[1, "s2_mean_15"], 8)
        self.assertAlmostEqual(final_cycle.loc[1, "s2_std_15"], np.std(range(1, 16), ddof=1))
        self.assertAlmostEqual(final_cycle.loc[2, "s2_mean_15"], 108)
        self.assertEqual(len(engineered), 30)

    def test_engineering_matches_existing_test_feature_data(self):
        processed = pd.read_csv("data/Processed Data/test_features.csv")
        raw_columns = [
            column
            for column in processed.columns
            if column in RAW_COLUMNS
        ]

        engineered = engineer_features(processed.loc[:, raw_columns])
        rolling_columns = [
            column
            for sensor in SENSOR_COLUMNS
            for column in (f"{sensor}_mean_15", f"{sensor}_std_15")
        ]
        expected = processed.sort_values(["unit", "cycle"]).reset_index(drop=True)

        pd.testing.assert_frame_equal(
            engineered.loc[:, rolling_columns],
            expected.loc[:, rolling_columns],
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_rejects_history_with_cycle_gaps(self):
        records = pd.DataFrame(
            [
                {column: 0 for column in RAW_COLUMNS}
                for _ in range(15)
            ]
        )
        records["unit"] = 1
        records["cycle"] = list(range(1, 15)) + [16]

        with self.assertRaisesRegex(ValueError, "consecutive"):
            engineer_features(records)


if __name__ == "__main__":
    unittest.main()
