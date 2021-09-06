import copy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from tifffile.tifffile import TIFF, TiffPageSeries
from wsidicom.geometry import Size, SizeMm
from wsidicom.interface import FileImporter, WsiGenericInstance

from ndpi_tiler import NdpiTiler


class NdpiFileImporter(FileImporter):
    def __init__(
        self,
        filepath: Path,
        base_dataset: Dataset,
        include_series: Dict[str, Tuple[int, Dict[int, Union[str, Dataset]]]],
        tile_size: Size,
        turbo_path: Path
    ):
        super().__init__(
            filepath,
            base_dataset,
            include_series,
            pydicom.uid.JPEGBaseline8Bit
        )
        self.tile_size = tile_size
        self.turbo_path = turbo_path
        self.tiler = NdpiTiler(
            self.filepath,
            self.tile_size,
            self.turbo_path
        )

    def close(self) -> None:
        self.tiler.close()

    @staticmethod
    def _get_mpp_from_page(page: TiffPageSeries) -> SizeMm:
        x_resolution = page.tags['XResolution'].value[0]
        y_resolution = page.tags['YResolution'].value[0]
        resolution_unit = page.tags['ResolutionUnit'].value
        if resolution_unit != TIFF.RESUNIT.CENTIMETER:
            raise ValueError("Unkown resolution unit")

        mpp_x = 1/x_resolution
        mpp_y = 1/y_resolution
        return Size(mpp_x, mpp_y)

    @staticmethod
    def _get_image_type(level_index: int) -> List[str]:
        if level_index == 0:
            return ['ORGINAL', 'PRIMARY', 'VOLUME', 'NONE']
        return ['DERIVED', 'PRIMARY', 'VOLUME', 'RESAMPLED']

    def _create_instance_dataset(
        self,
        level_index: int,
        level_uid: int,
        level: TiffPageSeries
    ) -> Dataset:
        dataset = copy.deepcopy(self.base_dataset)
        dataset.ImageType = self._get_image_type(level_index)
        dataset.SOPInstanceUID = level_uid
        shared_functional_group_sequence = Dataset()
        pixel_measure_sequence = Dataset()
        pixel_spacing = self._get_mpp_from_page(level.pages[0])

        pixel_measure_sequence.PixelSpacing = [
            pixel_spacing.width,
            pixel_spacing.height
        ]
        pixel_measure_sequence.SpacingBetweenSlices = 0.0
        pixel_measure_sequence.SliceThickness = 0.0
        shared_functional_group_sequence.PixelMeasuresSequence = (
            DicomSequence([pixel_measure_sequence])
        )
        dataset.SharedFunctionalGroupsSequence = DicomSequence(
            [shared_functional_group_sequence]
        )
        dataset.TotalPixelMatrixColumns = level.shape[1]
        dataset.TotalPixelMatrixRows = level.shape[0]
        dataset.Columns = self.tile_size[0]
        dataset.Rows = self.tile_size[1]
        dataset.ImagedVolumeWidth = (
            level.shape[1] * pixel_spacing.width
        )
        dataset.ImagedVolumeHeight = (
            level.shape[0] * pixel_spacing.height
        )
        dataset.ImagedVolumeDepth = 0.0
        dataset.SamplesPerPixel = 3
        dataset.PhotometricInterpretation = 'YBR_FULL_422'
        dataset.InstanceNumber = 0
        dataset.FocusMethod = 'AUTO'
        dataset.ExtendedDepthOfField = 'NO'
        return dataset

    def level_instances(self):
        instances = []

        series_definition = self.include_series['VOLUME']

        series_index = series_definition[0]
        for level_index, level_definition in series_definition[1].items():
            # If str, create dataset with str as uid
            # Otherwise use instance_dataset as dataset
            if isinstance(level_definition, str):
                instance_dataset = self._create_instance_dataset(
                    level_index,
                    level_definition,
                    self.tiler.series[series_index].levels[level_index]
                )
            elif isinstance(level_definition, Dataset):
                instance_dataset = level_definition
            else:
                raise ValueError()

            instance = WsiGenericInstance(
                self.tiler.get_level(series_index, level_index),
                instance_dataset,
                self.transfer_syntax
            )
            instances.append(instance)
        return instances

    def label_instances(self):
        return []

    def overview_instances(self):
        return []
