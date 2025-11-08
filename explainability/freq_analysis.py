"""
Frequency Analysis Module for XAI
Decompose and analyze frequency domain contributions to segmentation
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional
import cv2
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import seaborn as sns


class FrequencyDomainAnalyzer:
    """Analyze frequency domain characteristics of predictions and features"""
    
    def __init__(self, device: str = 'cuda', save_dir: str = 'results/figures/xai/freq'):
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_frequency_components(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract frequency components from image using FFT
        
        Args:
            image: Input image (H, W)
            
        Returns:
            Tuple of (low_freq, high_freq) components
        """
        # Apply FFT
        fft_image = fft2(image)
        fft_shift = fftshift(fft_image)
        
        # Create frequency magnitude
        freq_magnitude = np.abs(fft_shift)
        freq_phase = np.angle(fft_shift)
        
        # Create low-pass and high-pass filters
        h, w = image.shape
        rows, cols = np.ogrid[:h, :w]
        center_row, center_col = h // 2, w // 2
        
        # Distance from center
        radius_matrix = np.sqrt((rows - center_row)**2 + (cols - center_col)**2)
        
        # Threshold for low/high frequency separation
        cutoff = min(h, w) // 8
        
        # Low-frequency: central region
        low_freq_mask = radius_matrix <= cutoff
        low_freq = fft_shift * low_freq_mask
        
        # High-frequency: outer region
        high_freq_mask = radius_matrix > cutoff
        high_freq = fft_shift * high_freq_mask
        
        # Inverse FFT
        low_freq_spatial = np.abs(ifft2(ifftshift(low_freq)))
        high_freq_spatial = np.abs(ifft2(ifftshift(high_freq)))
        
        return low_freq_spatial, high_freq_spatial
    
    def compute_frequency_spectrum(self, image: np.ndarray) -> np.ndarray:
        """
        Compute frequency spectrum magnitude
        
        Args:
            image: Input image (H, W)
            
        Returns:
            Frequency spectrum (log scale)
        """
        fft_image = fft2(image)
        fft_shift = fftshift(fft_image)
        magnitude = np.abs(fft_shift)
        spectrum = np.log1p(magnitude)
        
        return spectrum
    
    def analyze_feature_map_frequency(self, feature_map: torch.Tensor) -> Dict:
        """
        Analyze frequency content of feature maps
        
        Args:
            feature_map: Feature tensor (C, H, W) or (B, C, H, W)
            
        Returns:
            Dictionary with frequency analysis
        """
        if len(feature_map.shape) == 4:
            # Take first sample and first channel
            feature_map = feature_map[0, 0].numpy()
        elif len(feature_map.shape) == 3:
            feature_map = feature_map[0].numpy()
        else:
            feature_map = feature_map.numpy()
        
        # Normalize
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        # Compute FFT
        fft_feat = fft2(feature_map)
        fft_shift = fftshift(fft_feat)
        magnitude = np.abs(fft_shift)
        
        # Compute power
        power = magnitude ** 2
        
        # Compute frequency band energies
        h, w = feature_map.shape
        rows, cols = np.ogrid[:h, :w]
        center_row, center_col = h // 2, w // 2
        radius_matrix = np.sqrt((rows - center_row)**2 + (cols - center_col)**2)
        
        # Define frequency bands
        r_max = np.sqrt(h**2 + w**2) / 2
        bands = {
            'very_low': (0, r_max * 0.1),
            'low': (r_max * 0.1, r_max * 0.2),
            'mid': (r_max * 0.2, r_max * 0.5),
            'high': (r_max * 0.5, r_max)
        }
        
        band_energies = {}
        for band_name, (r_min, r_max) in bands.items():
            mask = (radius_matrix >= r_min) & (radius_matrix < r_max)
            band_energies[band_name] = power[mask].sum()
        
        total_energy = power.sum()
        band_energies = {k: v / total_energy for k, v in band_energies.items()}
        
        return {
            'magnitude_spectrum': magnitude,
            'power': power,
            'band_energies': band_energies,
            'total_energy': total_energy
        }
    
    def visualize_frequency_spectrum(self, image: np.ndarray, output_path: Path):
        """
        Visualize frequency spectrum of image
        
        Args:
            image: Input image
            output_path: Save path
        """
        # Handle multi-channel image
        if len(image.shape) == 3:
            image = image.mean(axis=2)
        
        # Compute spectrum
        spectrum = self.compute_frequency_spectrum(image)
        
        # Decompose to LF and HF
        lf, hf = self.extract_frequency_components(image)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Frequency spectrum
        im1 = axes[0, 1].imshow(spectrum, cmap='hot')
        axes[0, 1].set_title('Frequency Spectrum (log)')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Radial frequency profile
        h, w = image.shape
        rows, cols = np.ogrid[:h, :w]
        center_row, center_col = h // 2, w // 2
        radius_matrix = np.sqrt((rows - center_row)**2 + (cols - center_col)**2)
        
        fft_image = fft2(image)
        fft_shift = fftshift(fft_image)
        magnitude = np.abs(fft_shift)
        
        # Compute radial profile
        max_radius = int(np.sqrt(h**2 + w**2) / 2)
        radial_profile = []
        for r in range(max_radius):
            mask = np.abs(radius_matrix - r) < 1
            radial_profile.append(magnitude[mask].mean())
        
        axes[0, 2].plot(radial_profile)
        axes[0, 2].set_xlabel('Frequency (radial)')
        axes[0, 2].set_ylabel('Magnitude')
        axes[0, 2].set_title('Radial Frequency Profile')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Low frequency component
        im3 = axes[1, 0].imshow(lf, cmap='gray')
        axes[1, 0].set_title('Low-Frequency Component')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # High frequency component
        im4 = axes[1, 1].imshow(hf, cmap='gray')
        axes[1, 1].set_title('High-Frequency Component')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1])
        
        # Reconstruction check
        reconstructed = lf + hf
        im5 = axes[1, 2].imshow(reconstructed, cmap='gray')
        axes[1, 2].set_title('LF + HF Reconstruction')
        axes[1, 2].axis('off')
        plt.colorbar(im5, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def compare_prediction_frequencies(self, lf_pred: np.ndarray,
                                      hf_pred: np.ndarray,
                                      full_pred: np.ndarray,
                                      ground_truth: np.ndarray,
                                      output_path: Path):
        """
        Compare frequency content of predictions
        
        Args:
            lf_pred: LF-only prediction
            hf_pred: HF-only prediction
            full_pred: Full model prediction
            ground_truth: Ground truth mask
            output_path: Save path
        """
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        predictions = {
            'LF-Only': lf_pred,
            'HF-Only': hf_pred,
            'Full': full_pred,
            'GT': ground_truth
        }
        
        for idx, (name, pred) in enumerate(predictions.items()):
            # Original prediction
            axes[0, idx].imshow(pred, cmap='Paired')
            axes[0, idx].set_title(f'{name} Prediction')
            axes[0, idx].axis('off')
            
            # Frequency spectrum
            spectrum = self.compute_frequency_spectrum(pred)
            im = axes[1, idx].imshow(spectrum, cmap='hot')
            axes[1, idx].set_title(f'{name} Spectrum')
            axes[1, idx].axis('off')
            plt.colorbar(im, ax=axes[1, idx])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def analyze_boundary_sharpness(self, prediction: np.ndarray,
                                  ground_truth: np.ndarray) -> Dict:
        """
        Analyze boundary sharpness using frequency analysis
        
        Args:
            prediction: Predicted segmentation mask
            ground_truth: Ground truth mask
            
        Returns:
            Dictionary with boundary sharpness metrics
        """
        # Extract boundaries using Sobel
        from scipy.ndimage import sobel
        
        pred_edges = np.hypot(
            sobel(prediction, axis=0),
            sobel(prediction, axis=1)
        )
        gt_edges = np.hypot(
            sobel(ground_truth, axis=0),
            sobel(ground_truth, axis=1)
        )
        
        # Compute frequency spectra
        pred_spectrum = self.compute_frequency_spectrum(pred_edges)
        gt_spectrum = self.compute_frequency_spectrum(gt_edges)
        
        # Compute high-frequency energy ratio
        h, w = prediction.shape
        rows, cols = np.ogrid[:h, :w]
        center_row, center_col = h // 2, w // 2
        radius_matrix = np.sqrt((rows - center_row)**2 + (cols - center_col)**2)
        
        r_max = np.sqrt(h**2 + w**2) / 2
        high_freq_mask = radius_matrix > (r_max * 0.5)
        
        pred_high_energy = pred_spectrum[high_freq_mask].sum()
        pred_total_energy = pred_spectrum.sum()
        
        gt_high_energy = gt_spectrum[high_freq_mask].sum()
        gt_total_energy = gt_spectrum.sum()
        
        return {
            'prediction_high_freq_ratio': pred_high_energy / (pred_total_energy + 1e-8),
            'gt_high_freq_ratio': gt_high_energy / (gt_total_energy + 1e-8),
            'prediction_edge_strength': pred_edges.sum(),
            'gt_edge_strength': gt_edges.sum(),
        }
    
    def create_frequency_band_visualization(self, image: np.ndarray, output_path: Path):
        """
        Create detailed frequency band decomposition visualization
        
        Args:
            image: Input image
            output_path: Save path
        """
        # Compute bands
        analysis = self.analyze_feature_map_frequency(torch.from_numpy(image.astype(np.float32)))
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Magnitude spectrum
        im1 = axes[0, 1].imshow(analysis['magnitude_spectrum'], cmap='log')
        axes[0, 1].set_title('Magnitude Spectrum')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Power spectrum
        im2 = axes[0, 2].imshow(np.log1p(analysis['power']), cmap='hot')
        axes[0, 2].set_title('Power Spectrum')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2])
        
        # Band energies bar chart
        bands = analysis['band_energies']
        axes[1, 0].bar(bands.keys(), bands.values(), color='steelblue', alpha=0.7)
        axes[1, 0].set_ylabel('Energy Ratio')
        axes[1, 0].set_title('Frequency Band Energies')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Cumulative energy
        cumulative = np.cumsum(list(bands.values()))
        axes[1, 1].plot(list(bands.keys()), cumulative, marker='o')
        axes[1, 1].fill_between(range(len(bands)), 0, cumulative, alpha=0.3)
        axes[1, 1].set_ylabel('Cumulative Energy')
        axes[1, 1].set_title('Cumulative Frequency Energy')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Summary text
        axes[1, 2].axis('off')
        summary_text = "Frequency Band Analysis\n\n"
        for band, energy in bands.items():
            summary_text += f"{band.capitalize()}: {energy:.3f}\n"
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


# Export classes
__all__ = ['FrequencyDomainAnalyzer']
