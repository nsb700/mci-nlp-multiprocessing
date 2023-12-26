import pytesseract
import torch.multiprocessing as mp
from PyPDF2 import PdfReader, PageObject
from pdf2image import convert_from_path
from PIL import Image
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import re
import logging
from pathlib import Path
import itertools
import timeit
from logging import config
import msiconstants
from wordcloud import WordCloud

config.dictConfig(msiconstants.LOG_CONFIG)
logger = logging.getLogger(__name__)

try:
    torch.multiprocessing.set_start_method('spawn', force=True)
except:
    pass

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract' 


def getIcdDict(filename: str):
    df = pd.read_csv(filename)
    df_dict = df.to_dict()
    code_to_index_dict = dict(zip(df['icdcode'], df.index))
    icdcodeslist = df['icdcode'].to_list()
    return df_dict, code_to_index_dict, icdcodeslist


def findGoodAndBadFiles(filepath: str, logs_queue):
    try:
        reader = PdfReader(filepath)
    except:
        logs_queue.put(f'Bad file: {filepath}')
        return ['bad', filepath, 0]
    page_count = len(reader.pages)
    first_page = reader.pages[0]
    first_page_text = first_page.extract_text()
    if len(first_page_text) <= 2:
        logs_queue.put(f'Good scanned file: {filepath}')
        return ['good_scanned', filepath, page_count]
    else:
        logs_queue.put(f'Good regular file: {filepath}')
        return ['good_regular', filepath, page_count]


def logs_queue_listener(logs_queue, logger):
    while 1:
        m = logs_queue.get()
        if m == 'kill':
            break
        logger.info(str(m))


def results_queue_listener(gbq, fn):
    with open(fn, 'w') as f:
        fileheader = "chartname,pagenumber,diagcodeLvl1,diagdescLvl1,pdftext"
        f.write(str(fileheader) + '\n')
        f.flush()
        while 1:
            m = gbq.get()
            if m == 'kill':
                break
            f.write(str(m) + '\n')
            f.flush()


def getNlpResults(just_chart_name, page_number, page_text_lines_string: str, df_dict, code_to_index_dict, icdcodeslist, model: SentenceTransformer, codes_file_embeddings, threshold, logs_queue, results_queue):
    logs_queue.put(f'Start nlp - chart: {just_chart_name}, page: {page_number}')
    page_text_lines = page_text_lines_string.split('\n')
    page_text_embeddings = model.encode(page_text_lines, convert_to_tensor=True, normalize_embeddings=True)
    res = []
    # Check for cosine similarity
    cosine_scores = util.dot_score(codes_file_embeddings, page_text_embeddings)
    idx = torch.nonzero(cosine_scores > threshold)
    for i in range(idx.shape[0]):
        idx_of_icdcode = idx[i][0].item()
        idx_of_pdftext = idx[i][1].item()
        diagcodelvl1 = df_dict['diagcodelvl1'][idx_of_icdcode]
        diagdesclvl1 = df_dict['diagdesclvl1'][idx_of_icdcode]
        ln = f"{just_chart_name},{page_number},{diagcodelvl1},{diagdesclvl1},{page_text_lines[idx_of_pdftext]}"
        res.append(ln)
    # Check if icdcode in pdftext
    crossprod = list(itertools.product(icdcodeslist, page_text_lines))
    code_in_text = [(pair[0], pair[1]) for pair in crossprod if pair[0] in pair[1]]
    for icdcode, pdftext in code_in_text:
        idx_of_icdcode = code_to_index_dict[icdcode]
        diagcodelvl1 = df_dict['diagcodelvl1'][idx_of_icdcode]
        diagdesclvl1 = df_dict['diagdesclvl1'][idx_of_icdcode]
        ln = f"{just_chart_name},{page_number},{diagcodelvl1},{diagdesclvl1},{pdftext}"
        res.append(ln)
    if len(res) > 0:
        res = list(set(res))
        results_queue.put('\n'.join(res))
        logs_queue.put(f'End nlp - chart: {just_chart_name}, page: {page_number}')
        return res
       

def readChartPageText(chart_name, page_number, page_object: PageObject, image_object: Image, logs_queue):
    just_chart_name = Path(chart_name).stem
    page_text = ''
    page_text_lines = []
    logs_queue.put(f'Start text read - chart: {just_chart_name}, page: {page_number}')
    if page_object is not None:  
        page_text = page_object.extract_text()
    elif image_object is not None:
        page_text = pytesseract.image_to_string(image_object)
    page_text = page_text.lower()
    linearr = page_text.split('\n')
    page_text = ''
    for line in linearr:
        x = re.sub(r'[^\w\s]', '', line.strip())
        if len(x) > 0:
            page_text_lines.append(x)
    logs_queue.put(f'End text read - chart: {just_chart_name}, page: {page_number}')
    return [just_chart_name, page_number, '\n'.join(page_text_lines)]


if __name__ == "__main__":

    start = timeit.default_timer()
    page_counter = 0
    logger.info('Start run')

    #==================================================================
    # Read transformed_codesfile, embeddings and create transformer model
    #================================================================== 
    df_dict, code_to_index_dict, icdcodeslist = getIcdDict('transformed_codesfile.csv')
    model = SentenceTransformer(model_name_or_path = msiconstants.MODEL_NAME)
    codes_file_embeddings = torch.load(f='tensor.pt', map_location='cpu')

    #================
    # Directory paths
    #================ 
    allfiles_directory  = Path.cwd()/'00_allfiles'
    errorfiles_directory = Path.cwd()/'01_errorfiles'
    finalresult_directory = Path.cwd()/'02_finalresult'

    #=========================
    # Paths for pfd charts
    #=========================
    input_file_paths = [Path(allfiles_directory)/f for f in allfiles_directory.iterdir()]
    
    #==========================================
    # Create process pool, queues and listeners
    #==========================================
    pool = mp.Pool(processes=msiconstants.PROCESS_COUNT)
    manager = mp.Manager()

    logs_queue = manager.Queue()    
    logs_queue_watcher = pool.apply_async(logs_queue_listener, args=(logs_queue, logger))
    
    results_queue = manager.Queue()
    final_result_filepath = Path(finalresult_directory)/'output.csv'
    results_queue_watcher = pool.apply_async(results_queue_listener, args=(results_queue, final_result_filepath))

    #===================================
    # Determine Good versus Bad Charts
    #===================================
    findGoodAndBadFiles_jobs = []
    for filepath in input_file_paths:
        job = pool.apply_async(findGoodAndBadFiles, args=(filepath, logs_queue))
        findGoodAndBadFiles_jobs.append(job)
    good_bad_files_res = []
    for job in findGoodAndBadFiles_jobs:
        good_bad_files_res.append(job.get())
    
    #=====================
    # Read Chart Page Text
    #=====================
    read_page_text_jobs = []
    for res in good_bad_files_res:
        flag, filepath, page_count  = res[0], res[1], res[2]
        if flag in ['good_regular', 'good_scanned']:
            if flag == 'good_regular':
                reader = PdfReader(filepath)
                pages = reader.pages
                page_counter += page_count
                for i in range(len(pages)):
                    job = pool.apply_async(readChartPageText, args=(filepath, i, pages[i], None, logs_queue))    
                    read_page_text_jobs.append(job) 
            elif flag == 'good_scanned':
                chunks = (page_count - 1) // msiconstants.BATCH_SIZE + 1
                page_counter += page_count
                for chc in range(chunks):
                    batch_page_from = chc*(msiconstants.BATCH_SIZE)
                    batch_page_to = (chc+1)*(msiconstants.BATCH_SIZE)
                    imgs = convert_from_path(filepath, first_page=batch_page_from, last_page=batch_page_to)
                    for i in range(len(imgs)):
                        p = batch_page_from+i+1 if chc==0 else batch_page_from+i
                        job = pool.apply_async(readChartPageText, args=(filepath, p, None, imgs[i], logs_queue))    
                        read_page_text_jobs.append(job) 
        elif flag == 'bad':
            filepath = Path(filepath)
            filebasename = filepath.name
            errorfilepath = Path(errorfiles_directory)/filebasename
            logger.info('Moving - oldpath: %s to newpath: %s', filepath, errorfilepath)
            filepath.rename(errorfilepath)
    page_text_results = []
    for job in read_page_text_jobs:
        page_text_results.append(job.get())

    #===================================
    # Get NLP Results
    #===================================
    nlp_result_jobs = []
    for tpl in page_text_results:
        just_chart_name = tpl[0]
        page_number = tpl[1]
        page_text_lines_string = tpl[2]
        job = pool.apply_async(getNlpResults, args=(just_chart_name, page_number, page_text_lines_string, df_dict, code_to_index_dict, icdcodeslist, model, codes_file_embeddings, msiconstants.THRESHOLD, logs_queue, results_queue))
        nlp_result_jobs.append(job) 
    nlp_results = []
    for job in nlp_result_jobs:
        nlp_results.append(job.get())

    #====================================================
    # Mark end of NLP run, kill results queue and close process pool
    #====================================================
    stop = timeit.default_timer()
    minutes_taken_to_run = (stop - start)/60
    logs_queue.put(f'End NLP run. Processed {page_counter} pages in {minutes_taken_to_run} minutes')
    results_queue.put('kill')

    #=======================================
    # WordCloud for each chart in output.csv
    #=======================================
    final_result_df = pd.read_csv(final_result_filepath)
    opbychart = [(chartname,df) for chartname,df in final_result_df.groupby(by=['chartname'])]
    for i in range(len(opbychart)):
        chartname = opbychart[i][0][0]
        df = opbychart[i][1]
        wordcloud = WordCloud(background_color="white", scale=1.5).generate(' '.join(list(df['diagdescLvl1'].unique())))
        wcpath = f'{finalresult_directory}/{chartname}.png'
        wordcloud.to_file(wcpath)
        logs_queue.put(f'WordCloud {wcpath} created')

    #========================================================
    # Mark end of run, kill logs queue and close process pool
    #========================================================
    logs_queue.put(f'End run')    
    logs_queue.put('kill')
    
    pool.close()
    pool.join()