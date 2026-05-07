#!/bin/bash
python test/test_device.py

AGENT_TYPE=$1

if [ -z "$AGENT_TYPE" ]; then
    echo "Error: please input agent type"
    exit 1
fi


case "$AGENT_TYPE" in
    "aerial-cactus-identification")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/aerial-cactus-identification/config_aerial-cactus-identification.yaml \
            --task ${DATA_ROOT}/aerial-cactus-identification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_aerial-cactus-identification.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "aptos2019-blindness-detection")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/aptos2019-blindness-detection/config_aptos2019-blindness-detection.yaml \
            --task ${DATA_ROOT}/aptos2019-blindness-detection/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_aptos2019-blindness-detection.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "denoising-dirty-documents")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/denoising-dirty-documents/config_denoising-dirty-documents.yaml \
            --task ${DATA_ROOT}/denoising-dirty-documents/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_denoising-dirty-documents.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "detecting-insults-in-social-commentary")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/detecting-insults-in-social-commentary/config_detecting-insults-in-social-commentary.yaml \
            --task ${DATA_ROOT}/detecting-insults-in-social-commentary/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_detecting-insults-in-social-commentary.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "dog-breed-identification")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/dog-breed-identification/config_dog-breed-identification.yaml \
            --task ${DATA_ROOT}/dog-breed-identification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_dog-breed-identification.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "dogs-vs-cats-redux-kernels-edition")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/dogs-vs-cats-redux-kernels-edition/config_dogs-vs-cats-redux-kernels-edition.yaml \
            --task ${DATA_ROOT}/dogs-vs-cats-redux-kernels-edition/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_dogs-vs-cats-redux-kernels-edition.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "histopathologic-cancer-detection")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/histopathologic-cancer-detection/config_histopathologic-cancer-detection.yaml \
            --task ${DATA_ROOT}/histopathologic-cancer-detection/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_histopathologic-cancer-detection.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "jigsaw-toxic-comment-classification-challenge")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/jigsaw-toxic-comment-classification-challenge/config_jigsaw-toxic-comment-classification-challenge.yaml \
            --task ${DATA_ROOT}/jigsaw-toxic-comment-classification-challenge/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/jigsaw-toxic-comment-classification-challenge/algo.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "leaf-classification")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/leaf-classification/config_leaf-classification.yaml \
            --task ${DATA_ROOT}/leaf-classification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_leaf-classification.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "mlsp-2013-birds")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/mlsp-2013-birds/config_mlsp-2013-birds.yaml \
            --task ${DATA_ROOT}/mlsp-2013-birds/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/mlsp-2013-birds/algo.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "new-york-city-taxi-fare-prediction")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/new-york-city-taxi-fare-prediction/config_new-york-city-taxi-fare-prediction.yaml \
            --task ${DATA_ROOT}/new-york-city-taxi-fare-prediction/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/new-york-city-taxi-fare-prediction/algo.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "nomad2018-predict-transparent-conductors")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/nomad2018-predict-transparent-conductors/config_nomad2018-predict-transparent-conductors.yaml \
            --task ${DATA_ROOT}/nomad2018-predict-transparent-conductors/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/nomad2018-predict-transparent-conductors/algo.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "plant-pathology-2020-fgvc7")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/plant-pathology-2020-fgvc7/config_plant-pathology-2020-fgvc7.yaml \
            --task ${DATA_ROOT}/plant-pathology-2020-fgvc7/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/plant-pathology-2020-fgvc7/data_loader.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "random-acts-of-pizza")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/random-acts-of-pizza/config_random-acts-of-pizza.yaml \
            --task ${DATA_ROOT}/random-acts-of-pizza/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_random-acts-of-pizza.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "ranzcr-clip-catheter-line-classification")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/ranzcr-clip-catheter-line-classification/config_ranzcr-clip-catheter-line-classification.yaml \
            --task ${DATA_ROOT}/ranzcr-clip-catheter-line-classification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_ranzcr-clip-catheter-line-classification.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "siim-isic-melanoma-classification")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/siim-isic-melanoma-classification/config_siim-isic-melanoma-classification.yaml \
            --task ${DATA_ROOT}/siim-isic-melanoma-classification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/siim-isic-melanoma-classification/data_loader.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "spooky-author-identification")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/spooky-author-identification/config_spooky-author-identification.yaml \
            --task ${DATA_ROOT}/spooky-author-identification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_spooky-author-identification.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "tabular-playground-series-dec-2021")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/tabular-playground-series-dec-2021/config_tabular-playground-series-dec-2021.yaml \
            --task ${DATA_ROOT}/tabular-playground-series-dec-2021/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_tabular-playground-series-dec-2021.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "tabular-playground-series-may-2022")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/tabular-playground-series-may-2022/config_tabular-playground-series-may-2022.yaml \
            --task ${DATA_ROOT}/tabular-playground-series-may-2022/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_tabular-playground-series-may-2022.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "text-normalization-challenge-english-language")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/text-normalization-challenge-english-language/config_text-normalization-challenge-english-language.yaml \
            --task ${DATA_ROOT}/text-normalization-challenge-english-language/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_text-normalization-challenge-english-language.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "text-normalization-challenge-russian-language")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/text-normalization-challenge-russian-language/config_text-normalization-challenge-russian-language.yaml \
            --task ${DATA_ROOT}/text-normalization-challenge-russian-language/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_text-normalization-challenge-russian-language.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "the-icml-2013-whale-challenge-right-whale-redux")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/the-icml-2013-whale-challenge-right-whale-redux/config_the-icml-2013-whale-challenge-right-whale-redux.yaml \
            --task ${DATA_ROOT}/the-icml-2013-whale-challenge-right-whale-redux/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_the-icml-2013-whale-challenge-right-whale-redux.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "AI4Code")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/AI4Code/config_AI4Code.yaml \
            --task ${DATA_ROOT}/AI4Code/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_AI4Code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "alaska2-image-steganalysis")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/alaska2-image-steganalysis/config_alaska2-image-steganalysis.yaml \
            --task ${DATA_ROOT}/alaska2-image-steganalysis/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_alaska2-image-steganalysis.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "billion-word-imputation")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/billion-word-imputation/config_billion-word-imputation.yaml \
            --task ${DATA_ROOT}/billion-word-imputation/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_billion-word-imputation.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "cassava-leaf-disease-classification")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/cassava-leaf-disease-classification/config_cassava-leaf-disease-classification.yaml \
            --task ${DATA_ROOT}/cassava-leaf-disease-classification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_cassava-leaf-disease-classification.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "cdiscount-image-classification-challenge")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/cdiscount-image-classification-challenge/config_cdiscount-image-classification-challenge.yaml \
            --task ${DATA_ROOT}/cdiscount-image-classification-challenge/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_cdiscount-image-classification-challenge.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "chaii-hindi-and-tamil-question-answering")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/chaii-hindi-and-tamil-question-answering/config_chaii-hindi-and-tamil-question-answering.yaml \
            --task ${DATA_ROOT}/chaii-hindi-and-tamil-question-answering/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_chaii-hindi-and-tamil-question-answering.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "champs-scalar-coupling")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/champs-scalar-coupling/config_champs-scalar-coupling.yaml \
            --task ${DATA_ROOT}/champs-scalar-coupling/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_champs-scalar-coupling.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "facebook-recruiting-iii-keyword-extraction")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/facebook-recruiting-iii-keyword-extraction/config_facebook-recruiting-iii-keyword-extraction.yaml \
            --task ${DATA_ROOT}/facebook-recruiting-iii-keyword-extraction/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_facebook-recruiting-iii-keyword-extraction.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "freesound-audio-tagging-2019")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/freesound-audio-tagging-2019/config_freesound-audio-tagging-2019.yaml \
            --task ${DATA_ROOT}/freesound-audio-tagging-2019/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_freesound-audio-tagging-2019.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "google-quest-challenge")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/google-quest-challenge/config_google-quest-challenge.yaml \
            --task ${DATA_ROOT}/google-quest-challenge/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_google-quest-challenge.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "h-and-m-personalized-fashion-recommendations")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/h-and-m-personalized-fashion-recommendations/config_h-and-m-personalized-fashion-recommendations.yaml \
            --task ${DATA_ROOT}/h-and-m-personalized-fashion-recommendations/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_h-and-m-personalized-fashion-recommendations.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "herbarium-2020-fgvc7")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/herbarium-2020-fgvc7/config_herbarium-2020-fgvc7.yaml \
            --task ${DATA_ROOT}/herbarium-2020-fgvc7/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_herbarium-2020-fgvc7.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "herbarium-2021-fgvc8")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/herbarium-2021-fgvc8/config_herbarium-2021-fgvc8.yaml \
            --task ${DATA_ROOT}/herbarium-2021-fgvc8/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_herbarium-2021-fgvc8.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "herbarium-2022-fgvc9")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/herbarium-2022-fgvc9/config_herbarium-2022-fgvc9.yaml \
            --task ${DATA_ROOT}/herbarium-2022-fgvc9/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_herbarium-2022-fgvc9.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "hotel-id-2021-fgvc8")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/hotel-id-2021-fgvc8/config_hotel-id-2021-fgvc8.yaml \
            --task ${DATA_ROOT}/hotel-id-2021-fgvc8/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_hotel-id-2021-fgvc8.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "hubmap-kidney-segmentation")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/hubmap-kidney-segmentation/config_hubmap-kidney-segmentation.yaml \
            --task ${DATA_ROOT}/hubmap-kidney-segmentation/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_hubmap-kidney-segmentation.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "icecube-neutrinos-in-deep-ice")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/icecube-neutrinos-in-deep-ice/config_icecube-neutrinos-in-deep-ice.yaml \
            --task ${DATA_ROOT}/icecube-neutrinos-in-deep-ice/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_icecube-neutrinos-in-deep-ice.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "imet-2020-fgvc7")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/imet-2020-fgvc7/config_imet-2020-fgvc7.yaml \
            --task ${DATA_ROOT}/imet-2020-fgvc7/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_imet-2020-fgvc7.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "inaturalist-2019-fgvc6")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/inaturalist-2019-fgvc6/config_inaturalist-2019-fgvc6.yaml \
            --task ${DATA_ROOT}/inaturalist-2019-fgvc6/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_inaturalist-2019-fgvc6.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "iwildcam-2020-fgvc7")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/iwildcam-2020-fgvc7/config_iwildcam-2020-fgvc7.yaml \
            --task ${DATA_ROOT}/iwildcam-2020-fgvc7/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_iwildcam-2020-fgvc7.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "jigsaw-unintended-bias-in-toxicity-classification")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/jigsaw-unintended-bias-in-toxicity-classification/config_jigsaw-unintended-bias-in-toxicity-classification.yaml \
            --task ${DATA_ROOT}/jigsaw-unintended-bias-in-toxicity-classification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_jigsaw-unintended-bias-in-toxicity-classification.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "kuzushiji-recognition")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/kuzushiji-recognition/config_kuzushiji-recognition.yaml \
            --task ${DATA_ROOT}/kuzushiji-recognition/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_kuzushiji-recognition.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "learning-agency-lab-automated-essay-scoring-2")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/learning-agency-lab-automated-essay-scoring-2/config_learning-agency-lab-automated-essay-scoring-2.yaml \
            --task ${DATA_ROOT}/learning-agency-lab-automated-essay-scoring-2/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_learning-agency-lab-automated-essay-scoring-2.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "lmsys-chatbot-arena")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/lmsys-chatbot-arena/config_lmsys-chatbot-arena.yaml \
            --task ${DATA_ROOT}/lmsys-chatbot-arena/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_lmsys-chatbot-arena.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "multi-modal-gesture-recognition")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/multi-modal-gesture-recognition/config_multi-modal-gesture-recognition.yaml \
            --task ${DATA_ROOT}/multi-modal-gesture-recognition/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_multi-modal-gesture-recognition.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "osic-pulmonary-fibrosis-progression")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/osic-pulmonary-fibrosis-progression/config_osic-pulmonary-fibrosis-progression.yaml \
            --task ${DATA_ROOT}/osic-pulmonary-fibrosis-progression/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_osic-pulmonary-fibrosis-progression.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "petfinder-pawpularity-score")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/petfinder-pawpularity-score/config_petfinder-pawpularity-score.yaml \
            --task ${DATA_ROOT}/petfinder-pawpularity-score/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_petfinder-pawpularity-score.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "plant-pathology-2021-fgvc8")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/plant-pathology-2021-fgvc8/config_plant-pathology-2021-fgvc8.yaml \
            --task ${DATA_ROOT}/plant-pathology-2021-fgvc8/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_plant-pathology-2021-fgvc8.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "seti-breakthrough-listen")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/seti-breakthrough-listen/config_seti-breakthrough-listen.yaml \
            --task ${DATA_ROOT}/seti-breakthrough-listen/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_seti-breakthrough-listen.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "statoil-iceberg-classifier-challenge")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/statoil-iceberg-classifier-challenge/config_statoil-iceberg-classifier-challenge.yaml \
            --task ${DATA_ROOT}/statoil-iceberg-classifier-challenge/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_statoil-iceberg-classifier-challenge.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "tensorflow-speech-recognition-challenge")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/tensorflow-speech-recognition-challenge/config_tensorflow-speech-recognition-challenge.yaml \
            --task ${DATA_ROOT}/tensorflow-speech-recognition-challenge/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_tensorflow-speech-recognition-challenge.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "tensorflow2-question-answering")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/tensorflow2-question-answering/config_tensorflow2-question-answering.yaml \
            --task ${DATA_ROOT}/tensorflow2-question-answering/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_tensorflow2-question-answering.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "tgs-salt-identification-challenge")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/tgs-salt-identification-challenge/config_tgs-salt-identification-challenge.yaml \
            --task ${DATA_ROOT}/tgs-salt-identification-challenge/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_tgs-salt-identification-challenge.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "tweet-sentiment-extraction")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/tweet-sentiment-extraction/config_tweet-sentiment-extraction.yaml \
            --task ${DATA_ROOT}/tweet-sentiment-extraction/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_tweet-sentiment-extraction.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "us-patent-phrase-to-phrase-matching")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/us-patent-phrase-to-phrase-matching/config_us-patent-phrase-to-phrase-matching.yaml \
            --task ${DATA_ROOT}/us-patent-phrase-to-phrase-matching/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_us-patent-phrase-to-phrase-matching.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "uw-madison-gi-tract-image-segmentation")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/uw-madison-gi-tract-image-segmentation/config_uw-madison-gi-tract-image-segmentation.yaml \
            --task ${DATA_ROOT}/uw-madison-gi-tract-image-segmentation/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_uw-madison-gi-tract-image-segmentation.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "ventilator-pressure-prediction")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/ventilator-pressure-prediction/config_ventilator-pressure-prediction.yaml \
            --task ${DATA_ROOT}/ventilator-pressure-prediction/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_ventilator-pressure-prediction.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "whale-categorization-playground")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/whale-categorization-playground/config_whale-categorization-playground.yaml \
            --task ${DATA_ROOT}/whale-categorization-playground/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_whale-categorization-playground.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "3d-object-detection-for-autonomous-vehicles")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/3d-object-detection-for-autonomous-vehicles/config_3d-object-detection-for-autonomous-vehicles.yaml \
            --task ${DATA_ROOT}/3d-object-detection-for-autonomous-vehicles/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_3d-object-detection-for-autonomous-vehicles.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "bms-molecular-translation")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/bms-molecular-translation/config_bms-molecular-translation.yaml \
            --task ${DATA_ROOT}/bms-molecular-translation/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_bms-molecular-translation.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "google-research-identify-contrails-reduce-global-warming")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/google-research-identify-contrails-reduce-global-warming/config_google-research-identify-contrails-reduce-global-warming.yaml \
            --task ${DATA_ROOT}/google-research-identify-contrails-reduce-global-warming/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_google-research-identify-contrails-reduce-global-warming.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "hms-harmful-brain-activity-classification")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/hms-harmful-brain-activity-classification/config_hms-harmful-brain-activity-classification.yaml \
            --task ${DATA_ROOT}/hms-harmful-brain-activity-classification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_hms-harmful-brain-activity-classification.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "iwildcam-2019-fgvc6")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/iwildcam-2019-fgvc6/config_iwildcam-2019-fgvc6.yaml \
            --task ${DATA_ROOT}/iwildcam-2019-fgvc6/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_iwildcam-2019-fgvc6.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "nfl-player-contact-detection")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/nfl-player-contact-detection/config_nfl-player-contact-detection.yaml \
            --task ${DATA_ROOT}/nfl-player-contact-detection/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_nfl-player-contact-detection.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "predict-volcanic-eruptions-ingv-oe")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/predict-volcanic-eruptions-ingv-oe/config_predict-volcanic-eruptions-ingv-oe.yaml \
            --task ${DATA_ROOT}/predict-volcanic-eruptions-ingv-oe/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_predict-volcanic-eruptions-ingv-oe.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "rsna-2022-cervical-spine-fracture-detection")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/rsna-2022-cervical-spine-fracture-detection/config_rsna-2022-cervical-spine-fracture-detection.yaml \
            --task ${DATA_ROOT}/rsna-2022-cervical-spine-fracture-detection/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_rsna-2022-cervical-spine-fracture-detection.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "rsna-breast-cancer-detection")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/rsna-breast-cancer-detection/config_rsna-breast-cancer-detection.yaml \
            --task ${DATA_ROOT}/rsna-breast-cancer-detection/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_rsna-breast-cancer-detection.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "rsna-miccai-brain-tumor-radiogenomic-classification")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/rsna-miccai-brain-tumor-radiogenomic-classification/config_rsna-miccai-brain-tumor-radiogenomic-classification.yaml \
            --task ${DATA_ROOT}/rsna-miccai-brain-tumor-radiogenomic-classification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_rsna-miccai-brain-tumor-radiogenomic-classification.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "siim-covid19-detection")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/siim-covid19-detection/config_siim-covid19-detection.yaml \
            --task ${DATA_ROOT}/siim-covid19-detection/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_siim-covid19-detection.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "smartphone-decimeter-2022")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/smartphone-decimeter-2022/config_smartphone-decimeter-2022.yaml \
            --task ${DATA_ROOT}/smartphone-decimeter-2022/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_smartphone-decimeter-2022.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "stanford-covid-vaccine")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/stanford-covid-vaccine/config_stanford-covid-vaccine.yaml \
            --task ${DATA_ROOT}/stanford-covid-vaccine/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_stanford-covid-vaccine.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-minimize
        ;;

    "vesuvius-challenge-ink-detection")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/vesuvius-challenge-ink-detection/config_vesuvius-challenge-ink-detection.yaml \
            --task ${DATA_ROOT}/vesuvius-challenge-ink-detection/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_vesuvius-challenge-ink-detection.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;

    "vinbigdata-chest-xray-abnormalities-detection")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/vinbigdata-chest-xray-abnormalities-detection/config_vinbigdata-chest-xray-abnormalities-detection.yaml \
            --task ${DATA_ROOT}/vinbigdata-chest-xray-abnormalities-detection/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/algoonly/algoonly_vinbigdata-chest-xray-abnormalities-detection.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. (But for current initial state, you can only use the raw data)" \
            --test-feedback \
            --force-maximize
        ;;
    *)
        echo "Error, not supported task type '$AGENT_TYPE' in MLE-Bench"
        exit 1
        ;;
esac
