//! Metal device detection and management.
//!
//! This module provides utilities for detecting and initializing Metal devices
//! on Apple Silicon, with fallback to CPU when Metal is unavailable.

use crate::error::{DeviceError, Result};
use candle_core::Device as CandleDevice;

/// A wrapper around Candle's Device with additional Metal-specific functionality.
#[derive(Debug, Clone)]
pub struct Device {
    inner: CandleDevice,
}

/// Information about the detected device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Type of device (Metal, CPU, etc.)
    pub device_type: DeviceType,
    /// Device index (for multi-GPU systems)
    pub index: usize,
    /// Whether Metal is available on this system
    pub metal_available: bool,
}

/// The type of compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// Metal GPU device (Apple Silicon)
    Metal,
    /// CPU fallback device
    Cpu,
}

impl Device {
    /// Creates a new Metal device with the specified index.
    ///
    /// On Apple Silicon, this will use the Metal backend for GPU acceleration.
    /// If Metal is not available, returns an error.
    ///
    /// # Arguments
    ///
    /// * `index` - Device index (usually 0 for single-GPU systems)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use metal_candle::backend::Device;
    ///
    /// let device = Device::new_metal(0)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError::MetalUnavailable`] if Metal is not available on the system.
    pub fn new_metal(index: usize) -> Result<Self> {
        match CandleDevice::new_metal(index) {
            Ok(inner) => Ok(Self { inner }),
            Err(e) => Err(DeviceError::MetalUnavailable {
                reason: format!("Failed to initialize Metal device {index}: {e}"),
            }
            .into()),
        }
    }

    /// Creates a new CPU device as a fallback.
    ///
    /// This is useful for testing or when Metal is not available.
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::backend::Device;
    ///
    /// let device = Device::new_cpu();
    /// ```
    #[must_use]
    pub fn new_cpu() -> Self {
        Self {
            inner: CandleDevice::Cpu,
        }
    }

    /// Attempts to create a Metal device, falling back to CPU if unavailable.
    ///
    /// This is the recommended way to create a device for most use cases,
    /// as it will use Metal when available but gracefully fall back to CPU.
    ///
    /// # Arguments
    ///
    /// * `index` - Preferred Metal device index (usually 0)
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::backend::Device;
    ///
    /// let device = Device::new_with_fallback(0);
    /// ```
    #[must_use]
    pub fn new_with_fallback(index: usize) -> Self {
        Self::new_metal(index).unwrap_or_else(|_| Self::new_cpu())
    }

    /// Returns the underlying Candle device.
    ///
    /// This is useful when you need to pass the device to Candle operations directly.
    #[must_use]
    pub const fn as_candle_device(&self) -> &CandleDevice {
        &self.inner
    }

    /// Consumes self and returns the underlying Candle device.
    #[must_use]
    pub fn into_candle_device(self) -> CandleDevice {
        self.inner
    }

    /// Returns whether this device is using Metal.
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::backend::Device;
    ///
    /// let device = Device::new_cpu();
    /// assert!(!device.is_metal());
    /// ```
    #[must_use]
    pub fn is_metal(&self) -> bool {
        matches!(self.inner, CandleDevice::Metal(_))
    }

    /// Returns whether this device is using CPU.
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::backend::Device;
    ///
    /// let device = Device::new_cpu();
    /// assert!(device.is_cpu());
    /// ```
    #[must_use]
    pub fn is_cpu(&self) -> bool {
        matches!(self.inner, CandleDevice::Cpu)
    }

    /// Returns information about this device.
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::backend::Device;
    ///
    /// let device = Device::new_cpu();
    /// let info = device.info();
    /// assert_eq!(info.device_type, metal_candle::backend::DeviceType::Cpu);
    /// ```
    #[must_use]
    pub fn info(&self) -> DeviceInfo {
        let (device_type, index) = match &self.inner {
            CandleDevice::Metal(_) => {
                // For Metal devices, we store the index but Candle doesn't expose it directly
                // For now, we assume index 0 (single GPU)
                (DeviceType::Metal, 0)
            }
            CandleDevice::Cpu | CandleDevice::Cuda(_) => (DeviceType::Cpu, 0),
        };

        DeviceInfo {
            device_type,
            index,
            metal_available: Self::is_metal_available(),
        }
    }

    /// Checks if Metal is available on this system.
    ///
    /// This is useful for detecting Apple Silicon vs. other platforms.
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::backend::Device;
    ///
    /// if Device::is_metal_available() {
    ///     println!("Running on Apple Silicon with Metal support!");
    /// }
    /// ```
    #[must_use]
    pub fn is_metal_available() -> bool {
        CandleDevice::new_metal(0).is_ok()
    }
}

impl From<CandleDevice> for Device {
    fn from(inner: CandleDevice) -> Self {
        Self { inner }
    }
}

impl From<Device> for CandleDevice {
    fn from(device: Device) -> Self {
        device.inner
    }
}

impl AsRef<CandleDevice> for Device {
    fn as_ref(&self) -> &CandleDevice {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_device_creation() {
        let device = Device::new_cpu();
        assert!(device.is_cpu());
        assert!(!device.is_metal());
    }

    #[test]
    fn test_device_with_fallback() {
        let device = Device::new_with_fallback(0);
        // Should succeed regardless of platform
        let info = device.info();
        assert!(info.device_type == DeviceType::Metal || info.device_type == DeviceType::Cpu);
    }

    #[test]
    fn test_device_with_fallback_on_metal_platform() {
        let device = Device::new_with_fallback(0);
        if Device::is_metal_available() {
            // On Apple Silicon, should use Metal
            assert!(device.is_metal());
            let info = device.info();
            assert_eq!(info.device_type, DeviceType::Metal);
            assert!(info.metal_available);
        } else {
            // On other platforms, should fall back to CPU
            assert!(device.is_cpu());
            let info = device.info();
            assert_eq!(info.device_type, DeviceType::Cpu);
            assert!(!info.metal_available);
        }
    }

    #[test]
    fn test_device_info() {
        let device = Device::new_cpu();
        let info = device.info();

        assert_eq!(info.device_type, DeviceType::Cpu);
        assert_eq!(info.index, 0);
    }

    #[test]
    fn test_cpu_device_info_fields() {
        let device = Device::new_cpu();
        let info = device.info();

        assert_eq!(info.device_type, DeviceType::Cpu);
        assert_eq!(info.index, 0);
        // metal_available depends on the platform
        assert_eq!(info.metal_available, Device::is_metal_available());
    }

    #[test]
    fn test_metal_availability_detection() {
        // This test just ensures the function doesn't panic
        let _available = Device::is_metal_available();
    }

    #[test]
    fn test_metal_availability_consistency() {
        // Test that metal availability is consistent
        let available1 = Device::is_metal_available();
        let available2 = Device::is_metal_available();
        assert_eq!(available1, available2);
    }

    #[test]
    fn test_device_conversions() {
        let candle_device = CandleDevice::Cpu;
        let device: Device = candle_device.into();
        assert!(device.is_cpu());

        let device = Device::new_cpu();
        let candle_device: CandleDevice = device.into();
        assert!(matches!(candle_device, CandleDevice::Cpu));
    }

    #[test]
    fn test_as_candle_device() {
        let device = Device::new_cpu();
        let candle_ref = device.as_candle_device();
        assert!(matches!(candle_ref, CandleDevice::Cpu));
    }

    #[test]
    fn test_as_ref_trait() {
        let device = Device::new_cpu();
        let candle_ref: &CandleDevice = device.as_ref();
        assert!(matches!(candle_ref, CandleDevice::Cpu));
    }

    #[test]
    fn test_into_candle_device() {
        let device = Device::new_cpu();
        let candle_device = device.into_candle_device();
        assert!(matches!(candle_device, CandleDevice::Cpu));
    }

    #[test]
    fn test_device_type_equality() {
        assert_eq!(DeviceType::Cpu, DeviceType::Cpu);
        assert_eq!(DeviceType::Metal, DeviceType::Metal);
        assert_ne!(DeviceType::Cpu, DeviceType::Metal);
    }

    #[test]
    fn test_device_clone() {
        let device1 = Device::new_cpu();
        let device2 = device1.clone();
        assert!(device2.is_cpu());
        assert_eq!(device1.is_cpu(), device2.is_cpu());
    }

    #[test]
    fn test_device_debug() {
        let device = Device::new_cpu();
        let debug_str = format!("{:?}", device);
        assert!(debug_str.contains("Device"));
    }

    // Only run this test on actual Apple Silicon
    #[test]
    #[cfg(target_os = "macos")]
    fn test_metal_device_creation_on_macos() {
        // Attempt to create Metal device - may fail on non-Apple Silicon Macs
        match Device::new_metal(0) {
            Ok(device) => {
                assert!(device.is_metal());
                assert!(!device.is_cpu());
                let info = device.info();
                assert_eq!(info.device_type, DeviceType::Metal);
                assert!(info.metal_available);
            }
            Err(e) => {
                // Not on Apple Silicon, which is fine
                assert!(!Device::is_metal_available());
                // Verify error message contains useful info
                let error_msg = e.to_string();
                assert!(error_msg.contains("Metal") || error_msg.contains("unavailable"));
            }
        }
    }

    #[test]
    fn test_metal_creation_behavior() {
        // Test that Metal creation follows expected behavior
        let result = Device::new_metal(0);
        let is_available = Device::is_metal_available();

        if is_available {
            // If Metal is available, creation should succeed
            assert!(result.is_ok());
            let device = result.unwrap();
            assert!(device.is_metal());
        } else {
            // If Metal is not available, should return error
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_device_info_debug() {
        let device = Device::new_cpu();
        let info = device.info();
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("DeviceInfo"));
    }

    #[test]
    fn test_device_type_debug() {
        let cpu_type = DeviceType::Cpu;
        let metal_type = DeviceType::Metal;

        let cpu_debug = format!("{:?}", cpu_type);
        let metal_debug = format!("{:?}", metal_type);

        assert!(cpu_debug.contains("Cpu"));
        assert!(metal_debug.contains("Metal"));
    }

    #[test]
    fn test_metal_error_on_non_metal_platform() {
        // On non-Metal platforms, this tests the error path
        // On Metal platforms, we test error formatting with a simulated error
        let result = Device::new_metal(0);

        if !Device::is_metal_available() {
            // On non-Metal platforms, should error
            assert!(result.is_err());
            let error = result.unwrap_err();
            let error_msg = error.to_string();
            // Error message should be informative
            assert!(
                error_msg.contains("Metal") || error_msg.contains("unavailable"),
                "Error message should be descriptive: {error_msg}"
            );
        }
        // On Metal platforms, index 0 succeeds (tested elsewhere)
    }

    #[test]
    fn test_from_candle_metal_device() {
        // Test From trait with a Metal device if available
        if Device::is_metal_available() {
            if let Ok(candle_metal) = CandleDevice::new_metal(0) {
                let device: Device = candle_metal.into();
                assert!(device.is_metal());

                let info = device.info();
                assert_eq!(info.device_type, DeviceType::Metal);
                assert!(info.metal_available);
            }
        }
    }

    #[test]
    fn test_device_info_metal_path() {
        // Ensure Metal device info returns correct type
        if Device::is_metal_available() {
            let device = Device::new_with_fallback(0);
            if device.is_metal() {
                let info = device.info();
                assert_eq!(info.device_type, DeviceType::Metal);
                assert_eq!(info.index, 0);
                assert!(info.metal_available);
            }
        }
    }

    // Test the error formatting by checking DeviceError construction
    #[test]
    fn test_device_error_formatting() {
        // Test that our error type formats correctly
        use crate::error::DeviceError;

        let error = DeviceError::MetalUnavailable {
            reason: "Test error message".to_string(),
        };

        let error_string = error.to_string();
        assert!(error_string.contains("Test error message"));
        assert!(error_string.contains("Metal"));
    }
}
