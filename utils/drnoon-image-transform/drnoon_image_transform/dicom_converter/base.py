import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers import gdcm_handler, numpy_handler, pylibjpeg_handler


class DicomConverter:
    """Single image transform class."""

    def __init__(self) -> None:
        """Initialize a dicom converter.

        Args:
            param: The parameters for the dicom converter.
        """

        pass

    def _set_pixel_data_handlers(self, transfer_syntax_uid: str) -> None:
        """Set appropriate pydicom pixel data handlers based on a Transfer Syntax UID.

        Reference: https://pydicom.github.io/pydicom/stable/old/image_data_handlers.html
                https://pydicom.github.io/pydicom/stable/reference/handlers.pixel_data.html#module-pydicom.pixel_data_handlers
        """

        # List of available handlers
        available_handlers = [
            h for h in [gdcm_handler, pylibjpeg_handler, numpy_handler] if h.is_available()
        ]

        # Use pylibjpeg_handler for JPEG Baseline (Process 1)
        if transfer_syntax_uid == "1.2.840.10008.1.2.4.50":
            primary_handler = pylibjpeg_handler

        # Use numpy_handler for Explicit VR Little Endian
        elif transfer_syntax_uid == "1.2.840.10008.1.2.1":
            primary_handler = numpy_handler

        # Use gdcm_handler for the rest
        else:
            primary_handler = gdcm_handler

        # Check if the primary handler is available
        if primary_handler not in available_handlers:
            error_message = (
                f"No suitable pixel data handler available for Transfer Syntax UID {transfer_syntax_uid}. "
                f"Required handler '{primary_handler.__name__}' is not available."
            )
            raise RuntimeError(error_message)

        # Reorder handlers to prioritize the primary handler
        handlers = [primary_handler] + [h for h in available_handlers if h != primary_handler]

        # Set handlers in pydicom config
        pydicom.config.pixel_data_handlers = handlers

    def _convert_dicom_to_image_array(self, dcm: pydicom.Dataset) -> np.ndarray:
        """Convert a pydicom Dataset to an RGB image array."""

        # Set appropriate pixel data handlers
        self._set_pixel_data_handlers(dcm.file_meta.TransferSyntaxUID)

        # Extract the pixel array from the DICOM dataset
        img = dcm.pixel_array

        # Check the photometric interpretation metadata
        photometric_interpretation = dcm.PhotometricInterpretation

        if photometric_interpretation == "PALETTE COLOR":
            # Convert pixel data using color palette
            lut = dcm.pixel_lut()
            img = lut[img]

        elif photometric_interpretation in ["YBR_FULL", "YBR_FULL_422"]:
            # Convert color space from YBR to RGB
            img = pydicom.pixel_data_handlers.convert_color_space(
                img, photometric_interpretation, "RGB"
            )

        elif photometric_interpretation in ["MONOCHROME1", "MONOCHROME2"]:
            # If it's a grayscale image, convert it to a 3-channel image replicating the single channel
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        elif photometric_interpretation == "RGB":
            # The image is already in RGB format, no additional conversion required
            pass

        else:
            raise ValueError(
                f"Unsupported photometric interpretation: {photometric_interpretation}"
            )

        return img

    def __call__(self, dcm: pydicom.Dataset) -> np.ndarray:
        """Convert dicom to image.

        Args:
            dcm: Dicom file for using convert.

        Returns:
            Converted image.
        """

        image = self._convert_dicom_to_image_array(dcm)
        return image
