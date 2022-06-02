import cv2
import fitz
import numpy
import os
import re

from deskew import determine_skew
from HocrParser import HocrParser
from PIL import Image
from random import sample



def compute_doc_angle(docs, min_n_page=5, max_angle=5):
    """ computes a documents text angle"""
    
    angles = []
    retries=0
    while (len(angles)<min_n_page) & (retries<2*min_n_page):
        page = sample(docs, 1)[0]
        angle = determine_skew(page)
        if abs(angle)<max_angle:
            angles += [angle]
        retries+=1
    
    if retries>2*min_n_page:
        angle = 0
    else:
        angle = numpy.median(angles)
    
    return angle

def rotate_image(image, angle):
    """ rotates a rgb image"""

    # rotating image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def rotate_docs(docs, min_n_page=5, max_angle=5):
    """ Rotates all pages of the document if needed."""
    
    # computes median angle needed to straighten document
    angle = compute_doc_angle(docs, min_n_page, max_angle)
    
    # if rotation is needed
    if angle!=0:
        
        # rotates each pages
        new_docs = []
        for doc in docs:
            new_docs += [rotate_image(doc, angle)]
    
    # else no rotation is needed, the initial images are returned
    else:
        new_docs = docs
        
    return new_docs

def pdf_to_array(PATH, file, zooming=3, min_n_page=5, max_angle=5):
    """ Converts pdf file to numpy array of RGB values.
    
    Parameters
    ----------
     - PATH, string: folder where to find .pdf file
     - file, string: .pdf file name
     - zooming, int, default 3: intensity of zooming x and y axes
     
    Returns
    ----------
     - doc_array, list(numpy.array): list of arrays of RGB uint8 for each page
    """
    
    # remove extension of file name
    file_without_extension = '.'.join(file.split('.')[:-1])
    
    # opening the pdf file document 
    doc = fitz.open(PATH+file_without_extension+'.pdf')

    # initiating arrays with scaling
    image_matrix = fitz.Matrix(fitz.Identity)
    image_matrix.preScale(zooming, zooming)

    # getting pixels from each page of the document
    doc = [page.getPixmap(alpha = False, matrix=image_matrix) for page in doc]

    # converting pixels to an array of RGB (0 to 255) values for each page of the document
    doc = [numpy.array(Image.frombytes('RGB', [pix.width, pix.height], pix.samples)) for pix in doc]
    
    # rotating pages if needed
    doc = rotate_docs(doc, min_n_page, max_angle)

    return doc


def array_to_ocr_xml(docs, model, thresh=0.05):
    """ Applies doctTR model on numpy array of RGB values.
    
    Parameters
    ----------
     - doc_array, list(numpy.array): list of arrays of RGB uint8 for each page
     - model, ocr_predictor: ocr model from doctr.models
     - thresh, float, default 0.05: confidence threshold for each word recognition
     
    Returns
    ----------
     - xml_outputs, list(tuple(string, xml.etree.ElementTree)): xml output from docTR
    """
    
    # ocr models prediction on each pages of the document
    ocred_docs = model(docs)

    # adding white spaces after each word
    for page in ocred_docs.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    word.value += ' '
                    if word.confidence < thresh:
                        word.value = ' '
    
    # exporting to xml
    xml_outputs = ocred_docs.export_as_xml()
    
    return xml_outputs

def xml_to_pdfa(PATH, file, doc_array, xml_outputs, tol=3e-3, file_reader=open):
    """ Converts xml outputs from docTR to searchable PDF/A file.
    
    Parameters
    ----------
     - PATH, string: folder where to save .pdf file
     - file, string: .pdf file name
     - doc_array, list(numpy.array): list of arrays of RGB uint8 for each page
     - xml_outputs, list(tuple(string, xml.etree.ElementTree)): xml output from docTR
     
    Returns
    ----------
     - pdfa_dict, dict: dictionnary of int page number keys and string text value 
    """    
    
    # remove extension of file name
    file_without_extension = '.'.join(file.split('.')[:-1])
    
    # init parser
    parser = HocrParser(tol=tol)
    
    # init merged pdf file
    merger = PdfFileMerger()
    
    # iterate through the xml outputs and images and export to pdf/a
    # the image is optional else you can set invisible_text=False and the text will be printed on a blank page
    for i, (xml, img) in enumerate(zip(xml_outputs, doc_array)):
        
        # accessing xml.etree.ElementTree.ElementTree object
        xml_element_tree = xml[1]
        
        # exporting the page to pdf/a file
        parser.export_pdfa(f'{PATH+file_without_extension+str(i)}.pdf', hocr=xml_element_tree, image=img)
        
        # adding the page to merged pdf/a file
        page_pdf = file_reader(f'{PATH+file_without_extension+str(i)}.pdf', 'rb')
        merger.append(page_pdf)
        page_pdf.close()
        
    # saving merged pdf/a file
    merger.write(f'{PATH+file_without_extension}.pdf')
    
    clean_folder([f'{PATH+file_without_extension+str(i)}.pdf' for i in range(len(xml_outputs))])
    
    # accessing merged pdf/a file
    pdfa_doc = fitz.open(f'{PATH+file_without_extension}.pdf')
    
    # converting pdf
    pdfa_dict = {k:v.get_text() for k,v in enumerate(pdfa_doc)}
    
    return pdfa_dict