from PythonScripts.utility import TestModelWithExampleGM
import os

FILE_PATH = os.path.join("Data", "GM_Example", "El-Centro.xlsx")
MODEL_NAME = "SeT_2_HS_768_Layer_12_Head_12_Epoch_20_Acc_0.92_F1_0.83_Model"



def main():
    TestModelWithExampleGM(file_path=FILE_PATH,
                           model_name=MODEL_NAME,
                           model_version="V2.0",
                           mask_mode=False,
                           padding_mode="front")

if __name__ == '__main__':
    main()

