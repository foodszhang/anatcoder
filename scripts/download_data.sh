#!/bin/bash
set -e

echo "=== AnatCoder data download helper ==="
echo "This script documents the expected data sources."
echo ""
echo "1) TotalSegmentator v2: https://doi.org/10.5281/zenodo.6802614"
echo "2) AMOS 2022: https://amos22.grand-challenge.org/"
echo ""
echo "Place downloaded files under:"
echo "  data/raw/totalsegmentator/<case_id>/"
echo "  data/raw/amos/<case_id>/"
echo ""
echo "Each case should include at least:"
echo "  ct.nii.gz"
echo "  seg.nii.gz"
echo ""
echo "Automated authenticated downloading is dataset-policy dependent and should be added later."
