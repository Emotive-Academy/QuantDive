#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# StockHealth simulator
# [StockHealth GitRepo](https://github.com/sidbannet/stockhealth)
#
# Copyright 2022 [Siddhartha Banerjee](mailto:sidban@uwalumni.com)
#

import numpy as np


class Price:
    """Validator for price."""

    def __init__(self, operator, value: float):
        """Instantiate the object."""
        self.operator = operator
        self.value = value

    def __call__(self, func):
        """Calling the class."""

        # noinspection PyPep8Naming,PyDecorator
        @classmethod
        def wrapper(
            cls,
            S: np.float64,
            K: np.float64,
            T: np.float64,
            r: np.float64,
            q: np.float64,
            sigma: np.float64,
        ):
            """Wrapper function that decorates the function."""
            return self.operator(
                self.value,
                func(cls, S=S, K=K, T=T, r=r, q=q, sigma=sigma),
            )
        return wrapper
