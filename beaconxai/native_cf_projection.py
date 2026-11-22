"""Evaluate native evolutionary candidates *inside* UCI HAR acceleration constraint.

The original MultiSubSpaCE evolutionary code is unchanged; the model and
outlier-score interfaces receive the nearest raw-L2 projection of every
proposed mask-derived candidate. The optimizer sees model scores and anomaly
scores of exactly the signals ultimately returned. This is a RESEARCH
VARIANT, not an unchanged MultiSubSpaCE method or a realism guarantee.
"""
from __future__ import annotations

import numpy as np

from .cf_physical_projection import project_uci_har_preserve_gravity
from .native_cf_budget import CountedCFModel


class UciGravityProjector:
    def __init__(self, original, train_mean, train_scale):
        self.original=np.asarray(original,dtype=float)
        self.mean=np.asarray(train_mean,dtype=float)
        self.scale=np.asarray(train_scale,dtype=float)
        if self.original.ndim!=2 or self.original.shape[1]!=9:
            raise ValueError("UCI HAR original [T,9] required")
        # Validate once in the same math and normalization used at runtime.
        project_uci_har_preserve_gravity(
            self.original,self.original,self.mean,self.scale)

    def __call__(self, inputs):
        inputs=np.asarray(inputs,dtype=float)
        single=inputs.ndim==2
        if single:
            inputs=inputs[None]
        if inputs.ndim!=3 or inputs.shape[1:]!=self.original.shape:
            raise ValueError("projector expects [N,T,9]")
        projected=np.stack([
            project_uci_har_preserve_gravity(
                self.original,x,self.mean,self.scale) for x in inputs
        ])
        return projected[0] if single else projected


class ProjectedCFModel(CountedCFModel):
    def __init__(self,predict_many,original,train_mean,train_scale,
                 n_classes,budget):
        self.projector=UciGravityProjector(
            original,train_mean,train_scale)
        super().__init__(
            predict_many,self.projector.original.shape,n_classes,budget)

    def predict(self,inputs):
        return super().predict(self.projector(inputs))

    def verified_result(self,proposed,target,source=None):
        return super().verified_result(
            self.projector(proposed),target,
            source=self.projector.original if source is None else source)


class ProjectedTrainingOutlier:
    def __init__(self,train_only_outlier,projector):
        self.outlier=train_only_outlier
        self.projector=projector

    def get_outlier_scores(self,inputs):
        return self.outlier.get_outlier_scores(
            self.projector(inputs))
