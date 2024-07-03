# 📔 DECOR: Improving Coherence in L2 English Writing with a Novel Benchmark for Incoherence Detection, Reasoning, and Rewriting
<img src="figures/decor_overview.png" width="100%">


This is the repository for DECOR, a novel benchmark that includes expert annotations for detecting incoherence in L2 English writing, identifying the underlying reasons, and rewriting the incoherent sentences. 

The figure above shows an example data point in DECOR, containing input context-sentence pairs, binary detection label, reason types for incoherence, and human rewrites.

In general, this repository offers:

1. The data format (CSV) for DECOR 📔
2. A supervised fine-tuning pipeline with task-specfic synthetic data
3. A standardized evaluation pipeline for all three tasks

## News
- [2024/06] 🔥 We release the preprint of DECOR. Read our [paper](https://arxiv.org/abs/2406.19650) for more details!

## Table of Contents
- [Downloading the DECOR benchmark](#downloading-the-decor-benchmark)
- [Environment settings](#environment-settings)
- [Supervised Fine-tuning pipelines](#supervised-fine-tuning-pipelines)
- [Evaluating on DECOR](#evaluating-on-decor)
- [Citation](#citation)
- [Questions](#questions)

## Downloading the DECOR benchmark 📔
We release the dev and test data for each task in DECOR 📔. They can be downloaded from the following links in the table:

<style>
  .center-title th {
    text-align: center;
  }
</style>

<table class="center-title"><thead>
  <tr>
    <th rowspan="2"><br>Incoherence <br>Detection</th>
    <th colspan="4">Incoherence reasoning</th>
    <th rowspan="2"><br>Incoherence<br>Rewriting</th>
  </tr>
  <tr>
    <th>Cohesion</th>
    <th>Consistency</th>
    <th>Relevance</th>
    <th>Other</th>
  </tr></thead>
<tbody>
  <tr>
    <td><a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/dev/binary_dev.csv">dev set</a>
    </td>
    <td><a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/dev/cohesion_dev.csv">dev set</a>
    </td>
    <td>
    <a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/dev/consistency_dev.csv">dev set</a></td>
    <td>
    <a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/dev/relevance_dev.csv">dev set</a></td>
    <td>
    <a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/dev/other_dev.csv">dev set</a></td>
    <td>
    <a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/dev/rewrite_541.csv">dev set</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/test/test_1355_clean.csv">test set</a></td>
    <td>
    <a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/test/cohesion_test.csv">test set</a></td>
    <td>
    <a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/test/consistency_test.csv">test set</a></td>
    <td>
    <a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/test/relevance_test.csv">test set</a></td>
    <td>
    <a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/test/other_test.csv">test set</a></td>
    <td><a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/test/test_rewrite_213_no_delete.csv">test set</a></td>
  </tr>
</tbody>
</table>
