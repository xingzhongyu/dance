Preprocessing Steps and Available Function Candidates
====================

This page lists the preprocessing functions included in each step.

Preprocessing Step: filter.gene
-----------

- :class:`dance.transforms.FilterGenesPercentile`
- :class:`dance.transforms.FilterGenesScanpyOrder`
- :class:`dance.transforms.FilterGenesPlaceHolder`

Preprocessing Step: filter.cell
-----------

- :class:`dance.transforms.FilterCellsScanpyOrder`
- :class:`dance.transforms.FilterCellsPlaceHolder`
- :class:`dance.transforms.FilterCellsCommonMod`

Preprocessing Step: normalize
---------

- :class:`dance.transforms.ColumnSumNormalize`
- :class:`dance.transforms.ScTransform`
- :class:`dance.transforms.Log1P`
- :class:`dance.transforms.NormalizeTotal`
- :class:`dance.transforms.NormalizeTotalLog1P`
- :class:`dance.transforms.tfidfTransform`
- :class:`dance.transforms.NormalizePlaceHolder`

Preprocessing Step: filter.gene(highly_variable)
-----------

- :class:`dance.transforms.FilterGenesTopK`
- :class:`dance.transforms.FilterGenesRegression`
- :class:`dance.transforms.FilterGenesMatch`
- :class:`dance.transforms.HighlyVariableGenesRawCount`
- :class:`dance.transforms.HighlyVariableGenesLogarithmizedByTopGenes`
- :class:`dance.transforms.HighlyVariableGenesLogarithmizedByMeanAndDisp`
- :class:`dance.transforms.FilterGenesNumberPlaceHolder`

Preprocessing Step: feature.cell
------------

- :class:`dance.transforms.CellPCA`
- :class:`dance.transforms.CellSVD`
- :class:`dance.transforms.CellSparsePCA`
- :class:`dance.transforms.WeightedFeaturePCA`
- :class:`dance.transforms.WeightedFeatureSVD`
- :class:`dance.transforms.GaussRandProjFeature`
- :class:`dance.transforms.FeatureCellPlaceHolder`