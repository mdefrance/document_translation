# original code https://github.com/mindee/notebooks/blob/main/doctr/export_as_pdfa.ipynb
import numpy as np
import PyPDF2
import re

from math import atan,cos, sin
from PIL import Image
from reportlab.lib.colors import black
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from typing import Dict, Optional, Tuple
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element

class HocrParser():

    def __init__(self, tol=3e-3):
        self.box_pattern = re.compile(r'bbox((\s+\d+){4})')
        self.baseline_pattern = re.compile(r'baseline((\s+[\d\.\-]+){2})')
        self.tol=tol

    def _element_coordinates(self, element: Element) -> Dict:
        """
        Returns a tuple containing the coordinates of the bounding box around
        an element
        """
        out = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
        if 'title' in element.attrib:
            matches = self.box_pattern.search(element.attrib['title'])
            if matches:
                coords = matches.group(1).split()
                out = {'x1': int(coords[0]), 'y1': int(
                    coords[1]), 'x2': int(coords[2]), 'y2': int(coords[3])}
        return out

    def _get_baseline(self, element: Element) -> Tuple[float, float]:
        """
        Returns a tuple containing the baseline slope and intercept.
        """
        if 'title' in element.attrib:
            matches = self.baseline_pattern.search(
                element.attrib['title']).group(1).split()
            if matches:
                return float(matches[0]), float(matches[1])
        return (0.0, 0.0)

    def _pt_from_pixel(self, pxl: Dict, dpi: int) -> Dict:
        """
        Returns the quantity in PDF units (pt) given quantity in pixels
        """
        pt = [(c / dpi * inch) for c in pxl.values()]
        return {'x1': pt[0], 'y1': pt[1], 'x2': pt[2], 'y2': pt[3]}

    def _get_element_text(self, element: Element) -> str:
        """
        Return the textual content of the element and its children
        """
        text = ''
        if element.text is not None:
            text += element.text
        for child in element:
            text += self._get_element_text(child)
        if element.tail is not None:
            text += element.tail
        return text

    def export_pdfa(self,
                    out_filename: str,
                    hocr: ET.ElementTree,
                    image: Optional[np.ndarray] = None,
                    fontname: str = "Times-Roman",
                    fontsize: int = 12,
                    invisible_text: bool = True,
                    dpi: int = 300):
        """
        Generates a PDF/A document from a hOCR document.
        """

        width, height = None, None
        # Get the image dimensions
        for div in hocr.findall(".//div[@class='ocr_page']"):
            coords = self._element_coordinates(div)
            pt_coords = self._pt_from_pixel(coords, dpi)
            width, height = pt_coords['x2'] - \
                pt_coords['x1'], pt_coords['y2'] - pt_coords['y1']
            # after catch break loop
            break
        if width is None or height is None:
            raise ValueError("Could not determine page size")

        pdf = Canvas(out_filename, pagesize=(width, height), pageCompression=1)

        span_elements = [element for element in hocr.iterfind(".//span")]

        # getting lines
        ocr_lines = [line for line in span_elements if 'class' in line.attrib and line.attrib['class'] == 'ocr_line' and line is not None]

        # sorting lines by y1 and then x1 with a tolerance to remove the two-column functionality
        sorted_lines = ocr_lines[:]
        if len(ocr_lines)>0:
            ocr_coords = [self._element_coordinates(line) for line in ocr_lines]
            ocr_y1 = [coord.get('y1') for coord in ocr_coords]
            ocr_x1 = [coord.get('x1') for coord in ocr_coords]
            ocr_xyz = np.array(list(zip(ocr_x1, ocr_y1, range(len(ocr_x1)))))
            ocr_order = tolerance_sort(ocr_xyz, height*self.tol)[:,2]
            sorted_lines = [ocr_lines[i] for i in ocr_order]

        # adding each line's text to the PDF
        for line in sorted_lines:

            # get information from xml
            pxl_line_coords = self._element_coordinates(line)
            line_box = self._pt_from_pixel(pxl_line_coords, dpi)

            # compute baseline
            slope, pxl_intercept = self._get_baseline(line)
            if abs(slope) < 0.005:
                slope = 0.0
            angle = atan(slope)
            cos_a, sin_a = cos(angle), sin(angle)
            intercept = pxl_intercept / dpi * inch
            baseline_y2 = height - (line_box['y2'] + intercept)

            # configure options
            text = pdf.beginText()
            text.setFont(fontname, fontsize)
            pdf.setFillColor(black)
            if invisible_text:
                text.setTextRenderMode(3)  # invisible text

            # transform overlayed text
            text.setTextTransform(
                cos_a, -sin_a, sin_a, cos_a, line_box['x1'], baseline_y2)

            elements = line.findall(".//span[@class='ocrx_word']")
            for elem in elements:
                elemtxt = self._get_element_text(elem)#.strip()
                # replace unsupported characters
                elemtxt = elemtxt.translate(str.maketrans(
                    {'ﬀ': 'ff', 'ﬃ': 'f‌f‌i', 'ﬄ': 'f‌f‌l', 'ﬁ': 'fi', 'ﬂ': 'fl'}))
                if not elemtxt:
                    continue

                # compute string width
                pxl_coords = self._element_coordinates(elem)
                box = self._pt_from_pixel(pxl_coords, dpi)
                box_width = box['x2'] - box['x1']
                font_width = pdf.stringWidth(elemtxt, fontname, fontsize)

                # Adjust relative position of cursor
                cursor = text.getStartOfLine()
                dx = box['x1'] - cursor[0]
                dy = baseline_y2 - cursor[1]
                text.moveCursor(dx, dy)

                # suppress text if it is 0 units wide
                if font_width > 0:
                    text.setHorizScale(100 * box_width / font_width)
                    text.textOut(elemtxt)
            pdf.drawText(text)

        # overlay image if provided
        if image is not None:
            pdf.drawImage(ImageReader(Image.fromarray(image)),
                          0, 0, width=width, height=height)
        pdf.save()

def tolerance_sort(array, tolerance):
    array_sorted = np.copy(array[np.lexsort((array[:, 0], array[:, 1]))])
    sort_range = [0]
    for i in range(array.shape[0] - 1):
        if array_sorted[i + 1, 1] - array_sorted[i, 1] <= tolerance:
            sort_range.append(i + 1)
            continue
        else:
            sub_arr = np.take(array_sorted, sort_range, axis=0)
            sub_arr_ord = np.copy(
                sub_arr[np.lexsort((sub_arr[:, 1], sub_arr[:, 0]))])
            array_sorted[slice(sort_range[0], sort_range[-1] +
                               1)] = sub_arr_ord
            sort_range = [i + 1]
    return array_sorted