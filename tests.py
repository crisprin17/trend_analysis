import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from generate_synthetic_data import generate_synthetic_data
from trend_analysis import analyze_trend, calculate_loess_trend, log_likelihood_ratio_test

class TestTrendAnalysis(unittest.TestCase):
    def setUp(self):
        # Generate test data for each test
        self.upward_data = generate_synthetic_data('2023-01-01', 14, 'upward', noise_level=0.1)
        self.downward_data = generate_synthetic_data('2023-01-01', 14, 'downward', noise_level=0.1)
        self.no_trend_data = generate_synthetic_data('2023-01-01', 14, 'no_trend', noise_level=0.1)

    def test_minimum_data_requirement(self):
        """Test that analysis requires at least two weeks of data"""
        insufficient_data = self.upward_data.iloc[:7]  # Only one week
        with self.assertRaises(ValueError):
            analyze_trend(insufficient_data)

    def test_loess_trend_detection(self):
        """Test LOESS trend detection for different trend types"""
        # Test upward trend
        trend = calculate_loess_trend(self.upward_data)
        self.assertEqual(trend, "Upward")

        # Test downward trend
        trend = calculate_loess_trend(self.downward_data)
        self.assertEqual(trend, "Downward")

        # Test no trend
        trend = calculate_loess_trend(self.no_trend_data)
        self.assertEqual(trend, "No Trend")

    def test_llr_trend_detection(self):
        """Test Log-Likelihood Ratio trend detection"""
        # Test upward trend
        trend = log_likelihood_ratio_test(self.upward_data)
        self.assertEqual(trend, "Upward")

        # Test downward trend
        trend = log_likelihood_ratio_test(self.downward_data)
        self.assertEqual(trend, "Downward")

    def test_method_agreement(self):
        """Test that the analysis correctly identifies when methods agree/disagree"""
        # Test with clear upward trend
        results = analyze_trend(self.upward_data)
        self.assertTrue(results['methods_agree'].iloc[0])
        self.assertEqual(results['final_trend'].iloc[0], "Upward")

        # Test with clear downward trend
        results = analyze_trend(self.downward_data)
        self.assertTrue(results['methods_agree'].iloc[0])
        self.assertEqual(results['final_trend'].iloc[0], "Downward")

if __name__ == '__main__':
    unittest.main()
