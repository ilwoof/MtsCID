#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 30/04/2024 11:32 am

@author : Yongzheng Xie
@email : ilwoof@gmail.com
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_time_series_comparison(raw_data1, raw_data2, label, title_name="Time Series Comparison", highlight_label=1):
    B, L, C = raw_data1.shape

    # Convert tensors to numpy arrays if they are not already
    if isinstance(raw_data1, torch.Tensor):
        raw_data1 = raw_data1.numpy()
    if isinstance(raw_data2, torch.Tensor):
        raw_data2 = raw_data2.numpy()
    if isinstance(label, torch.Tensor):
        label = label.numpy()

    # Number of batches to plot in one figure
    num_batches_per_fig = 2
    num_figs = (B + num_batches_per_fig - 1) // num_batches_per_fig  # Calculate number of figures needed

    for fig_idx in range(num_figs):
        plt.figure(figsize=(12, 6))

        # Determine the range of batches to plot in this figure
        start_batch = fig_idx * num_batches_per_fig
        end_batch = min(start_batch + num_batches_per_fig, B)

        for batch_idx in range(start_batch, end_batch):
            if np.any(label[batch_idx] == highlight_label):
                info = 'with anomalies'
            else:
                info = 'without anomalies'
            # Create subplot for the first time series
            ax1 = plt.subplot(num_batches_per_fig, 2, (batch_idx - start_batch) * 2 + 1)
            if True:
                # Plot all variables for the first time series dataset
                for i in range(C):
                    ax1.plot(raw_data1[batch_idx, :, i], label=f'Var {i + 1} TS1')

                # Find segments where the label is `highlight_label`
                current_label = label[batch_idx]
                segments = []
                in_segment = False
                start_idx = None

                for idx in range(L):
                    if current_label[idx] == highlight_label:
                        if not in_segment:
                            start_idx = idx
                            in_segment = True
                    else:
                        if in_segment:
                            segments.append((start_idx, idx - 1))
                            in_segment = False

                # Handle case where the last segment goes to the end
                if in_segment:
                    segments.append((start_idx, L - 1))

                # Plot vertical lines at the beginning and end of each segment
                for start, end in segments:
                    ax1.axvline(x=start, color='red', linestyle='--', linewidth=0.8,
                                label='Label Start' if start == segments[0][0] else "")
                    ax1.axvline(x=end, color='red', linestyle='--', linewidth=0.8,
                                label='Label End' if end == segments[-1][1] else "")

            ax1.set_xlabel('Time')
            ax1.set_ylabel('Value')
            ax1.set_title(f'Original Time Series - Batch {batch_idx} {info}')
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

            # Create subplot for the second time series
            ax2 = plt.subplot(num_batches_per_fig, 2, (batch_idx - start_batch) * 2 + 2)
            # if np.any(label[batch_idx] == highlight_label):
            if True:
                # Plot all variables for the second time series dataset
                for i in range(C):
                    ax2.plot(raw_data2[batch_idx, :, i], label=f'Var {i + 1} TS2')

                # Find segments where the label is `highlight_label`
                current_label = label[batch_idx]
                segments = []
                in_segment = False
                start_idx = None

                for idx in range(L):
                    if current_label[idx] == highlight_label:
                        if not in_segment:
                            start_idx = idx
                            in_segment = True
                    else:
                        if in_segment:
                            segments.append((start_idx, idx - 1))
                            in_segment = False

                # Handle case where the last segment goes to the end
                if in_segment:
                    segments.append((start_idx, L - 1))

                # Plot vertical lines at the beginning and end of each segment
                for start, end in segments:
                    ax2.axvline(x=start, color='red', linestyle='--', linewidth=0.8,
                                label='Label Start' if start == segments[0][0] else "")
                    ax2.axvline(x=end, color='red', linestyle='--', linewidth=0.8,
                                label='Label End' if end == segments[-1][1] else "")

            ax2.set_xlabel('Time')
            ax2.set_ylabel('Value')
            ax2.set_title(f'Representation Time Series - Batch {batch_idx} {info}')
            # ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=8)

        plt.suptitle(title_name)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make space for the main title and legends
        plt.show()


def plot_mts_with_labels(raw_data, label, title_name="", highlight_label=1):
    B, L, C = raw_data.shape

    # Convert tensors to numpy arrays if they are not already
    if isinstance(raw_data, torch.Tensor):
        raw_data = raw_data.numpy()
    if isinstance(label, torch.Tensor):
        label = label.numpy()

    # Number of batches to plot in one figure
    num_batches_per_fig = 2
    num_figs = (B + num_batches_per_fig - 1) // num_batches_per_fig  # Calculate number of figures needed

    for fig_idx in range(num_figs):
        plt.figure(figsize=(12, 6))

        # Determine the range of batches to plot in this figure
        start_batch = fig_idx * num_batches_per_fig
        end_batch = min(start_batch + num_batches_per_fig, B)

        for batch_idx in range(start_batch, end_batch):
            ax = plt.subplot(1, num_batches_per_fig, batch_idx - start_batch + 1)

            # Check if the batch contains any timesteps where the label is `highlight_label`
            # if np.any(label[batch_idx] == highlight_label):
            if True:
                if np.any(label[batch_idx] == highlight_label):
                    info = 'with anomalies'
                else:
                    info = 'without anomalies'

                # Plot all variables for the current batch
                for i in range(C):
                    ax.plot(raw_data[batch_idx, :, i], label=f'Variable {i + 1}')

                # Find segments where the label is `highlight_label`
                current_label = label[batch_idx]
                segments = []
                in_segment = False
                start_idx = None

                for idx in range(L):
                    if current_label[idx] == highlight_label:
                        if not in_segment:
                            start_idx = idx
                            in_segment = True
                    else:
                        if in_segment:
                            segments.append((start_idx, idx - 1))
                            in_segment = False

                # Handle case where the last segment goes to the end
                if in_segment:
                    segments.append((start_idx, L - 1))

                # Plot vertical lines at the beginning and end of each segment
                for start, end in segments:
                    ax.axvline(x=start, color='red', linestyle='--', linewidth=0.8,
                               label='Label Start' if start == segments[0][0] else "")
                    ax.axvline(x=end, color='red', linestyle='--', linewidth=0.8,
                               label='Label End' if end == segments[-1][1] else "")

                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.set_title(f'Batch {batch_idx} {info}')
                # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=8)

        plt.suptitle(title_name)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make space for the main title and legends
        plt.show()
