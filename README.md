[![Publish Docker Image](https://github.com/Emotive-Academy/QuantDive/actions/workflows/docker.yaml/badge.svg)](https://github.com/Emotive-Academy/QuantDive/actions/workflows/docker.yaml)
[![Unit test CI](https://github.com/Emotive-Academy/QuantDive/actions/workflows/test.yaml/badge.svg)](https://github.com/Emotive-Academy/QuantDive/actions/workflows/test.yaml)

**Disclaimer**: This software is for educational purposes only. It is not intended to provide investment advice. Do not make investment decisions based on this software.

# Intro 
A powerful tool diving deep into the world of quantitative finance, providing robust analysis and insights for stock market enthusiasts and professionals alike. This tool is made available by [Emotive Academy](www.emotive.academy).

## Overview

This is Python-based tool that helps to develop and test trading strategies. It provides a set of tools for data analysis, strategy development, backtesting, and optimization. It is designed to be used by analysts and data scientists and educators.

- `stockhealth`: A Python package to model and analyze stock data.
- `utilities`: A Python package to provide utility functions for data analysis and visualization.

# Models

The models used in this project are:

- **[Heston Stochastic Model](https://en.wikipedia.org/wiki/Heston_model) for spot pricing**: 
    - The Heston stochastic volatility model is a model for the dynamics of the spot price of an asset. It is used to model the volatility of the spot price of an asset. The model is based on the assumption that the spot price of an asset follows a log-normal distribution, and that the volatility of the spot price is mean reverting.
- **[Black-Scholes](https://en.wikipedia.org/wiki/Black–Scholes_model) for option pricing**: 
    - The Black-Scholes model is a model for the pricing of European options. It is used to model the price of a European option on a non-dividend paying stock. The model is based on the assumption that the stock price follows a geometric Brownian motion, and that the option price is a function of the stock price, the strike price, the time to expiration, the risk-free interest rate, and the volatility of the stock price.

## Heston Stochastic Model

The basic Heston model assumes that $S_t$ is the spot price of the stock at time $t$, and that the volatility of the stock price is given by the square root of the variance process $v_t$. The variance process $v_t$ is assumed to follow a mean-reverting process, and is given by the following stochastic differential equation:

$$  dv_t = \kappa(\theta - v_t)dt + \sigma\sqrt{v_t}dW_t^v $$

where, 
* $\kappa$ is the mean-reversion rate
* $\theta$ is the long-term average variance
* $\sigma$ is the volatility of the variance process
* $W_t^v$ is a [Wiener process](https://en.wikipedia.org/wiki/Wiener_process) of volatility.

The stochastic process for the stock price $S_t$ is given by the following stochastic differential equation:

$$ dS_t = rS_tdt + \sqrt{v_t}S_tdW_t^S $$

where, 
* $r$ is the risk-free interest rate
* $W_t^S$ is a [Wiener process](https://en.wikipedia.org/wiki/Wiener_process) of the stock price.

such that covariance between $W_t^S$ and $W_t^v$ is $\rho$ and $dW_t^S dW_t^v = \rho dt$. Feller condition is $\kappa\theta > \sigma^2$ which ensures that the variance process $v_t$ is non-negative.

## Black-Scholes Model

The Black-Scholes-Merton model is a mathematical model for the dynamics of a financial market containing derivative securities. The model assumes that the price of the underlying asset follows a geometric Brownian motion, and that the risk-free interest rate is constant. The model is used to price European options on non-dividend paying stocks.

The governing partial differential equation for the price of a European call option is given by the Black-Scholes-Merton equation:

$$ \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2S^2\frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0 $$

where,
* $V$ is the price of the option
* $S$ is the price of the underlying stock
* $t$ is the time to expiration
* $r$ is the risk-free interest rate
* $\sigma$ is the volatility of the stock price

The key financial insight behind the equation is that one can perfectly [hedge](https://en.wikipedia.org/wiki/Hedge_(finance)) the option by buying and selling the underlying asset in just the right way and consequently eliminate risk.

# Outro

This project is part of the Emotive Academy's curriculum. It is designed to be used by students and educators to learn about quantitative finance and algorithmic trading. It is not intended to be used for real trading.

## Contributors

- [Siddhartha Banerjee](https://github.com/sidbannet)