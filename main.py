import argparse
from transformers import pipeline
from pyabsa import (
    AspectTermExtraction as ATEPC,
    DeviceTypeOption,
)
from pyabsa import TaskCodeOption
import csv
import json
import os

def json_file_writer(file_name, data):
    """
    Function for dump json file
    args:
        file_name -> (str): name of the file.
        data -> (list or dict): data for dump
    """
    with open(file_name, "w") as json_f:
        json.dump(data, json_f, indent=4, ensure_ascii=False)

def extraction(responses):
    """
    Extract the codes
    args:
        responses -> (list): list of the responses to open-ended question
    """
    
    codes = set()
    
    aspect_extractor = ATEPC.AspectExtractor('english', auto_device=DeviceTypeOption.AUTO)
    
    for response in responses:
        res = aspect_extractor.predict(response)
        codes.update(set(res['aspect']))
    
    codes = list(codes)

    print("Number of Codes:", len(codes))

    with open('codes.csv', 'w') as csvfile:
        print("writing...")

        writer = csv.writer(csvfile)
        writer.writerow(["code"])
        for code in codes:
            writer.writerow([code])

    print("Codes generation completed")
    print("The file is saved at {}/codes.csv".format(os.getcwd()))


def classification(responses, codes):
    """
    Codes classification
    args: 
        responses -> (list): list of the responses to open-ended question
        codes -> (list): list of the codes.
    """

    result_by_code = dict()
    result_by_response = list()
    stat_result = list()

    model_name = "EleutherAI/gpt-neo-1.3B"
    classifier = pipeline("zero-shot-classification", model=model_name)

    results = classifier(responses, codes)
    
    threshold = 1/len(codes)

    for result in results:
        scores = result['scores']
        labels = result['labels']
        text = result['sequence']
        
        tmp_dict_response = {
                    "response": text,
                    "codes": []
                }
        for idx in range(len(scores)):
            if scores[idx] > threshold:
                tmp_score = scores[idx]
                tmp_code = codes[idx]

                if tmp_code not in result_by_code:
                    result_by_code[tmp_code] = {
                            "number": 0,
                            "responses": []
                        }


                result_by_code[tmp_code]["number"] += 1
                result_by_code[tmp_code]["responses"].append(
                            {
                                "response": text,
                                "confidence": tmp_score
                            }
                        )
                
                tmp_dict_response["codes"].append(
                            {
                                "code": tmp_code,
                                "confidence": tmp_score
                            }
                        )

        result_by_response.append(tmp_dict_response)
    for i in list(result_by_code.keys()):
        stat_result.append(
                    {
                        "code": i,
                        "number": result_by_code[i]["number"]
                    }
                )
    
    json_file_writer("result_by_responses.json", result_by_response)
    json_file_writer("result_by_codes.json", result_by_code)
    json_file_writer("stat_result.json", stat_result)

    print("Results are saved as: result_by_responses.json, result_by_codes.json, and stat_result.json")






def main():
    parser = argparse.ArgumentParser(
        description="A tool for code extraciotn and classification. Use -h to show help message."
        )
    parser.add_argument("--file_path")
    parser.add_argument("--extraction", action="store_true")
    parser.add_argument("--classification", action="store_true")
    #parser.add_argument("--threshold", default=0.3, type=float)
    args = parser.parse_args()


    input_responses_path = args.file_path
    responses = []

    with open(input_responses_path) as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            responses.append(row["response"])

    if args.extraction:
        extraction(responses)

    if args.classification:
        codes = []
        with open("codes.csv") as csvfile:
            rows = csv.DictReader(csvfile)
            for row in rows:
                codes.append(row["code"])
        classification(responses, codes)

    print("Completed!")

    
main()
