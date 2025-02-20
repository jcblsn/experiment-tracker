# experiment-tracker

## Overview

A very lightweight alternative to MLFlow's experiment tracking capabilities.

Last year, I was using MLFlow for a complex modeling project at work and was surprised to find it slow to a halt after only a few hundred runs. This repo was inspired by my subsequent research about alternatives, especially Eduardo Blancas' [Who needs MLflow when you have SQLite?](https://ploomber.io/blog/experiment-tracking/).

The system uses a SQLite backend for direct SQL queries, operates locally without a server, and has no dependencies outside the standard library--making it preferable for many solo projects where simplicity and control are a priority.

## Installation

Requires Python 3.12 or later.

## Usage

The main interface is exposed through the `ExperimentTracker` class. A typical workflow follows:

1. Create an experiment
2. Start a run under that experiment
3. Log model details and parameters
4. Log predictions, which will automatically compute metrics
5. End the run

Examples are provided in `src.main.py`.

## Schema

The database contains tables for experiments, runs, models, predictions, and metrics.

## Testing

Use `python -m unittest discover tests` to run the test suite.