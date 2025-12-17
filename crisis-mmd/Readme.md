# Crisis MMD

The project builds on python 3.10.0.

1) Install requirements: `pip install -r requirements.txt`. (Create a virtual env if preferred, and then install dependencies within the environment.)

2) Run `python app_refactored.py` and go to `http://127.0.0.1:5000`.

You might need access to a few files/folders not on the repository that you can find in the dropbox [link](https://www.dropbox.com/scl/fo/kg6zguy0rsxy6hl3nfgnq/AK8bqN95Mx5_4vJaU3DeH0k?rlkey=4zyow700k0t2cdyj1vx9ulswg&e=1&st=9smixcj4&dl=0). 
- `data_dump/all_images_data_dump.npy`
- `data_dump/dmd_images_data_dump.npy`
- `data_image`
- `dmd`
- `incidents_1m`
- `model` (contains all the trained models)
- `env` (might not be necessary after installing the requirements with pip)

For incidents numpy batches can generate it with the `prepare_incidents1m_dump.py` or download the batches from data_dump (batched) - [data_dump_batches.zip](https://www.dropbox.com/scl/fi/54yo7darp29jbfykd8ytw/data_dump_batches.zip?rlkey=5awvp8hsj0lk1jedw4p08jeho&st=x6fkmdzw&dl=0) or
[incidents_data_dump.npy.zip](https://www.dropbox.com/scl/fi/he1dabnyalz7cn1t91b5x/incidents_images_data_dump.npy.zip?rlkey=pqteafikuuq3b4pd1fvwif8bs&st=861221bz&dl=0).
## Tree structure of the project

A snapshot of local working directory, to organize the folder structure.

```
.
├── Readme.md
├── __pycache__
│   ├── aidrtokenize.cpython-310.pyc
│   ├── aidrtokenize.cpython-37.pyc
│   ├── aidrtokenize.cpython-38.pyc
│   ├── aidrtokenize.cpython-39.pyc
│   ├── app_refactored.cpython-310.pyc
│   ├── crisis_data_generator_image_optimized.cpython-310.pyc
│   ├── crisis_data_generator_image_optimized.cpython-37.pyc
│   ├── crisis_data_generator_image_optimized.cpython-38.pyc
│   ├── crisis_data_generator_image_optimized.cpython-39.pyc
│   ├── data_process_multimodal_pair.cpython-310.pyc
│   ├── data_process_multimodal_pair.cpython-37.pyc
│   ├── data_process_multimodal_pair.cpython-38.pyc
│   ├── data_process_multimodal_pair.cpython-39.pyc
│   ├── data_process_new.cpython-310.pyc
│   └── data_process_new.cpython-38.pyc
├── aidrtokenize.py
├── app.py
├── app_refactored.py
├── crisis_data_generator_image_optimized.py
├── crisis_mmd
│   ├── bin
│   ├── include
│   ├── lib
│   ├── pyvenv.cfg
│   └── share
├── data_dump
│   ├── all_images_data_dump.npy
│   ├── dmd_images_data_dump.npy
│   ├── incidents1m_images_data_dump.npy
│   ├── incidents_images_data_batch_0.npy
│   ├── incidents_images_data_batch_1.npy
│   ├── incidents_images_data_batch_10.npy
│   ├── incidents_images_data_batch_11.npy
│   ├── incidents_images_data_batch_12.npy
│   ├── incidents_images_data_batch_13.npy
│   ├── incidents_images_data_batch_14.npy
│   ├── incidents_images_data_batch_15.npy
│   ├── incidents_images_data_batch_16.npy
│   ├── incidents_images_data_batch_17.npy
│   ├── incidents_images_data_batch_18.npy
│   ├── incidents_images_data_batch_19.npy
│   ├── incidents_images_data_batch_2.npy
│   ├── incidents_images_data_batch_20.npy
│   ├── incidents_images_data_batch_21.npy
│   ├── incidents_images_data_batch_22.npy
│   ├── incidents_images_data_batch_23.npy
│   ├── incidents_images_data_batch_24.npy
│   ├── incidents_images_data_batch_25.npy
│   ├── incidents_images_data_batch_26.npy
│   ├── incidents_images_data_batch_27.npy
│   ├── incidents_images_data_batch_28.npy
│   ├── incidents_images_data_batch_29.npy
│   ├── incidents_images_data_batch_3.npy
│   ├── incidents_images_data_batch_30.npy
│   ├── incidents_images_data_batch_31.npy
│   ├── incidents_images_data_batch_32.npy
│   ├── incidents_images_data_batch_33.npy
│   ├── incidents_images_data_batch_34.npy
│   ├── incidents_images_data_batch_35.npy
│   ├── incidents_images_data_batch_36.npy
│   ├── incidents_images_data_batch_37.npy
│   ├── incidents_images_data_batch_38.npy
│   ├── incidents_images_data_batch_39.npy
│   ├── incidents_images_data_batch_4.npy
│   ├── incidents_images_data_batch_40.npy
│   ├── incidents_images_data_batch_41.npy
│   ├── incidents_images_data_batch_42.npy
│   ├── incidents_images_data_batch_43.npy
│   ├── incidents_images_data_batch_44.npy
│   ├── incidents_images_data_batch_45.npy
│   ├── incidents_images_data_batch_46.npy
│   ├── incidents_images_data_batch_47.npy
│   ├── incidents_images_data_batch_48.npy
│   ├── incidents_images_data_batch_49.npy
│   ├── incidents_images_data_batch_5.npy
│   ├── incidents_images_data_batch_50.npy
│   ├── incidents_images_data_batch_51.npy
│   ├── incidents_images_data_batch_52.npy
│   ├── incidents_images_data_batch_53.npy
│   ├── incidents_images_data_batch_54.npy
│   ├── incidents_images_data_batch_55.npy
│   ├── incidents_images_data_batch_56.npy
│   ├── incidents_images_data_batch_57.npy
│   ├── incidents_images_data_batch_58.npy
│   ├── incidents_images_data_batch_59.npy
│   ├── incidents_images_data_batch_6.npy
│   ├── incidents_images_data_batch_60.npy
│   ├── incidents_images_data_batch_61.npy
│   ├── incidents_images_data_batch_62.npy
│   ├── incidents_images_data_batch_63.npy
│   ├── incidents_images_data_batch_64.npy
│   ├── incidents_images_data_batch_65.npy
│   ├── incidents_images_data_batch_66.npy
│   ├── incidents_images_data_batch_67.npy
│   ├── incidents_images_data_batch_68.npy
│   ├── incidents_images_data_batch_69.npy
│   ├── incidents_images_data_batch_7.npy
│   ├── incidents_images_data_batch_70.npy
│   ├── incidents_images_data_batch_71.npy
│   ├── incidents_images_data_batch_72.npy
│   ├── incidents_images_data_batch_73.npy
│   ├── incidents_images_data_batch_8.npy
│   └── incidents_images_data_batch_9.npy
├── data_image
│   ├── california_wildfires
│   ├── hurricane_harvey
│   ├── hurricane_irma
│   ├── hurricane_maria
│   ├── iraq_iran_earthquake
│   ├── mexico_earthquake
│   └── srilanka_floods
├── data_process_multimodal_pair.py
├── data_process_new.py
├── dmd
│   ├── dmd_metadata.csv
│   ├── multimodal
│   └── readme.txt
├── dmd_inference.py
├── dmd_metadata.py
├── env
│   ├── Include
│   ├── Lib
│   ├── Scripts
│   └── pyvenv.cfg
├── generate_crisis_small_file.py
├── generate_dmd_small_file.py
├── hurricane_data
│   ├── Beryl
│   ├── Helene
│   ├── Kirk
│   ├── Milton
│   └── Rafael
├── incidents_1m
│   ├── incidents_metadata.csv
│   └── multimodal
├── metadata
│   ├── task_humanitarian_text_img_agreed_lab_dev.tsv
│   ├── task_humanitarian_text_img_agreed_lab_test.tsv
│   ├── task_humanitarian_text_img_lab_test.tsv
│   ├── task_informative_text_img_agreed_lab_dev.tsv
│   ├── task_informative_text_img_agreed_lab_test.tsv
│   ├── task_informative_text_img_lab_test.tsv
│   └── task_severity_test.tsv
├── misc
│   ├── cleanup_dmd.py
│   ├── dump_stats.py
│   ├── prepare_dmd_dump.py
│   ├── rough.py
│   └── verify.py
├── model
│   ├── hum_multimodal_paired_agreed_lab.tokenizer
│   ├── humanitarian_cnn_keras_09-04-2022_05-10-03.hdf5
│   ├── humanitarian_cnn_keras_09-04-2022_05-10-03.tokenizer
│   ├── humanitarian_image_vgg16_ferda.hdf5
│   ├── info_multimodal_paired_agreed_lab.tokenizer
│   ├── informative_image.hdf5
│   ├── informativeness_cnn_keras.hdf5
│   ├── informativeness_cnn_keras_09-04-2022_04-26-49.hdf5
│   ├── informativeness_cnn_keras_09-04-2022_04-26-49.tokenizer
│   ├── model_info_x copy.hdf5
│   ├── model_info_x.hdf5
│   ├── model_info_x1.hdf5
│   ├── model_info_x2.hdf5
│   ├── model_severe_x.hdf5
│   ├── model_severe_x1.hdf5
│   ├── model_severe_x2.hdf5
│   ├── model_x.hdf5
│   ├── model_x1.hdf5
│   ├── model_x2.hdf5
│   ├── severity_cnn_keras_21-07-2022_08-14-32.hdf5
│   ├── severity_cnn_keras_21-07-2022_08-14-32.tokenizer
│   └── severity_image.hdf5
├── performance_measures
│   ├── humanitarian.csv
│   ├── informative.csv
│   └── severity.csv
├── prepare_incidents1m_dump.py
├── requirements.txt
├── static
│   ├── Capture.PNG
│   ├── Capture1.PNG
│   ├── base.html
│   ├── bg.jpg
│   ├── bg.png
│   ├── capture3.PNG
│   ├── case1.png
│   ├── case1_1.PNG
│   ├── case1_Arch.PNG
│   ├── case2.png
│   ├── case2_2.PNG
│   ├── case2_arch.PNG
│   ├── case3.png
│   ├── case3_3.PNG
│   ├── case3_arch.png
│   ├── classification.PNG
│   ├── classification1.png
│   ├── css
│   ├── data_image
│   ├── data_image_wrong
│   ├── dmd
│   ├── favicon.ico
│   ├── heatmap.jpg
│   ├── humanitarian.csv
│   ├── incidents_1m
│   ├── index.html
│   ├── informative.csv
│   ├── js
│   ├── result.html
│   ├── severity.csv
│   ├── text-removebg-preview.png
│   ├── text.jpg
│   ├── text_heatmap.jpg
│   ├── uploads
│   ├── vgg-removebg-preview.png
│   ├── visualize.html
│   └── visualize.jpg
├── stop_words
│   └── stop_words_english.txt
├── templates
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   ├── temp
│   ├── user_data.html
│   ├── user_data_result.html
│   └── visualize.html
└── utils.py
```
