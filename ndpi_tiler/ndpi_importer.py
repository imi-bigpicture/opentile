from pathlib import Path
from typing import Dict, List, Tuple, Union
import copy

from tifffile import TiffFile
from tifffile.tifffile import TIFF, TiffPageSeries

from wsidicom.interface import WsiGenericInstance, FileImporter
from wsidicom.geometry import Size, SizeMm

from ndpi_tiler import NdpiTiler, NdpiFileHandle
from pydicom.uid import UID as Uid
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence


class NdpiFileImporter(FileImporter):
    def __init__(
        self,
        filepath: Path,
        base_dataset: Dataset,
        include_series: Dict[str, Tuple[int, List[Union[str, Dataset]]]],
        transfer_syntax: Uid,
        tile_size: Size,
        turbo_path: Path
    ):
        super().__init__(
            filepath,
            base_dataset,
            include_series,
            transfer_syntax
        )
        self.tile_size = tile_size
        self.turbo_path = turbo_path

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

    def level_instances(
        self,
        selected_levels: List[int] = None
    ):
        instances = []
        slide = TiffFile(self.filepath)
        filehandle = NdpiFileHandle(slide.filehandle)

        level_definition = self.include_series['VOLUME']
        level_series = level_definition[0]
        series: TiffPageSeries = slide.series[level_series]

        tiler = NdpiTiler(series, filehandle, self.tile_size, self.turbo_path)

        if selected_levels is None:
            levels_to_parse = {
                level_index: level
                for level_index, level in enumerate(series.levels)
            }
        else:
            levels_to_parse = {
                level_index: level
                for level_index, level in enumerate(series.levels)
                if level_index in selected_levels
            }
        for level_index, level in levels_to_parse.items():
            level_shape = level.shape

            pixel_spacing = self._get_mpp_from_page(level.pages[0])

            instance_ds = copy.deepcopy(self.base_dataset)
            instance_ds.ImageType = self._get_image_type(level_index)
            instance_ds.SOPInstanceUID = level_definition[1][level_index]
            shared_functional_group_sequence = Dataset()
            pixel_measure_sequence = Dataset()
            pixel_measure_sequence.PixelSpacing = [
                pixel_spacing.width,
                pixel_spacing.height
            ]
            pixel_measure_sequence.SpacingBetweenSlices = 0.0
            pixel_measure_sequence.SliceThickness = 0.0
            shared_functional_group_sequence.PixelMeasuresSequence = (
                DicomSequence([pixel_measure_sequence])
            )
            instance_ds.SharedFunctionalGroupsSequence = DicomSequence(
                [shared_functional_group_sequence]
            )
            instance_ds.TotalPixelMatrixColumns = level_shape[1]
            instance_ds.TotalPixelMatrixRows = level_shape[0]
            instance_ds.Columns = self.tile_size[0]
            instance_ds.Rows = self.tile_size[1]
            instance_ds.ImagedVolumeWidth = (
                level_shape[1] * pixel_spacing.width
            )
            instance_ds.ImagedVolumeHeight = (
                level_shape[0] * pixel_spacing.height
            )
            instance_ds.ImagedVolumeDepth = 0.0
            instance_ds.SamplesPerPixel = 3
            instance_ds.PhotometricInterpretation = 'YBR_FULL_422'
            instance_ds.InstanceNumber = 0
            instance_ds.FocusMethod = 'AUTO'
            instance_ds.ExtendedDepthOfField = 'NO'

            instance = WsiGenericInstance(
                tiler.get_level(level_index),
                instance_ds,
                self.transfer_syntax
            )
            instances.append(instance)
        return instances

    def label_instances(self):
        return []

    def overview_instances(self):
        return []
