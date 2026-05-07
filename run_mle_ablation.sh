#!/bin/bash
set -uo pipefail

TIME_LIMIT_SECS=43200   # 12h
KILL_AFTER="5m"

AGENT_TYPE=${1:-}

if [ -z "$AGENT_TYPE" ]; then
    echo "Error: please input agent type"
    exit 1
fi

FEISHU_WEBHOOK="${FEISHU_WEBHOOK:-}"

send_feishu() {
    local msg="$1"

    if [ -z "$FEISHU_WEBHOOK" ]; then
        return 0
    fi

    curl -s -X POST "$FEISHU_WEBHOOK" \
      -H 'Content-Type: application/json' \
      -d "{\"msg_type\":\"text\",\"content\":{\"text\":\"$msg\"}}" >/dev/null 2>&1 || true
}

echo "[start] ablation task=${AGENT_TYPE}, limit=${TIME_LIMIT_SECS}s"

timeout -k "$KILL_AFTER" "${TIME_LIMIT_SECS}s" bash -c '
set -uo pipefail

python test/test_device.py

AGENT_TYPE="$1"

case "$AGENT_TYPE" in
    "aerial-cactus-identification")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/aerial-cactus-identification/config_aerial-cactus-identification.yaml \
            --task ${DATA_ROOT}/aerial-cactus-identification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/aerial-cactus-identification/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "aptos2019-blindness-detection")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/aptos2019-blindness-detection/config_aptos2019-blindness-detection.yaml \
            --task ${DATA_ROOT}/aptos2019-blindness-detection/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/aptos2019-blindness-detection/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "denoising-dirty-documents")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/denoising-dirty-documents/config_denoising-dirty-documents.yaml \
            --task ${DATA_ROOT}/denoising-dirty-documents/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/denoising-dirty-documents/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "detecting-insults-in-social-commentary")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/detecting-insults-in-social-commentary/config_detecting-insults-in-social-commentary.yaml \
            --task ${DATA_ROOT}/detecting-insults-in-social-commentary/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/detecting-insults-in-social-commentary/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "dog-breed-identification")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/dog-breed-identification/config_dog-breed-identification.yaml \
            --task ${DATA_ROOT}/dog-breed-identification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/dog-breed-identification/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "dogs-vs-cats-redux-kernels-edition")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/dogs-vs-cats-redux-kernels-edition/config_dogs-vs-cats-redux-kernels-edition.yaml \
            --task ${DATA_ROOT}/dogs-vs-cats-redux-kernels-edition/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/dogs-vs-cats-redux-kernels-edition/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "histopathologic-cancer-detection")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/histopathologic-cancer-detection/config_histopathologic-cancer-detection.yaml \
            --task ${DATA_ROOT}/histopathologic-cancer-detection/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/histopathologic-cancer-detection/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "jigsaw-toxic-comment-classification-challenge")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/jigsaw-toxic-comment-classification-challenge/config_jigsaw-toxic-comment-classification-challenge.yaml \
            --task ${DATA_ROOT}/jigsaw-toxic-comment-classification-challenge/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/jigsaw-toxic-comment-classification-challenge/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "leaf-classification")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/leaf-classification/config_leaf-classification.yaml \
            --task ${DATA_ROOT}/leaf-classification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/leaf-classification/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "mlsp-2013-birds")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/mlsp-2013-birds/config_mlsp-2013-birds.yaml \
            --task ${DATA_ROOT}/mlsp-2013-birds/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/mlsp-2013-birds/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "new-york-city-taxi-fare-prediction")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/new-york-city-taxi-fare-prediction/config_new-york-city-taxi-fare-prediction.yaml \
            --task ${DATA_ROOT}/new-york-city-taxi-fare-prediction/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/new-york-city-taxi-fare-prediction/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "nomad2018-predict-transparent-conductors")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/nomad2018-predict-transparent-conductors/config_nomad2018-predict-transparent-conductors.yaml \
            --task ${DATA_ROOT}/nomad2018-predict-transparent-conductors/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/nomad2018-predict-transparent-conductors/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "plant-pathology-2020-fgvc7")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/plant-pathology-2020-fgvc7/config_plant-pathology-2020-fgvc7.yaml \
            --task ${DATA_ROOT}/plant-pathology-2020-fgvc7/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/plant-pathology-2020-fgvc7/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "random-acts-of-pizza")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/random-acts-of-pizza/config_random-acts-of-pizza.yaml \
            --task ${DATA_ROOT}/random-acts-of-pizza/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/random-acts-of-pizza/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "ranzcr-clip-catheter-line-classification")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/ranzcr-clip-catheter-line-classification/config_ranzcr-clip-catheter-line-classification.yaml \
            --task ${DATA_ROOT}/ranzcr-clip-catheter-line-classification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/ranzcr-clip-catheter-line-classification/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "siim-isic-melanoma-classification")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/siim-isic-melanoma-classification/config_siim-isic-melanoma-classification.yaml \
            --task ${DATA_ROOT}/siim-isic-melanoma-classification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/siim-isic-melanoma-classification/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "spooky-author-identification")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/spooky-author-identification/config_spooky-author-identification.yaml \
            --task ${DATA_ROOT}/spooky-author-identification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/spooky-author-identification/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "tabular-playground-series-dec-2021")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/tabular-playground-series-dec-2021/config_tabular-playground-series-dec-2021.yaml \
            --task ${DATA_ROOT}/tabular-playground-series-dec-2021/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/tabular-playground-series-dec-2021/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "tabular-playground-series-may-2022")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/tabular-playground-series-may-2022/config_tabular-playground-series-may-2022.yaml \
            --task ${DATA_ROOT}/tabular-playground-series-may-2022/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/tabular-playground-series-may-2022/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "text-normalization-challenge-english-language")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/text-normalization-challenge-english-language/config_text-normalization-challenge-english-language.yaml \
            --task ${DATA_ROOT}/text-normalization-challenge-english-language/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/text-normalization-challenge-english-language/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "text-normalization-challenge-russian-language")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/text-normalization-challenge-russian-language/config_text-normalization-challenge-russian-language.yaml \
            --task ${DATA_ROOT}/text-normalization-challenge-russian-language/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/text-normalization-challenge-russian-language/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "the-icml-2013-whale-challenge-right-whale-redux")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/the-icml-2013-whale-challenge-right-whale-redux/config_the-icml-2013-whale-challenge-right-whale-redux.yaml \
            --task ${DATA_ROOT}/the-icml-2013-whale-challenge-right-whale-redux/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/the-icml-2013-whale-challenge-right-whale-redux/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "AI4Code")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/AI4Code/config_AI4Code.yaml \
            --task ${DATA_ROOT}/AI4Code/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/AI4Code/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "alaska2-image-steganalysis")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/alaska2-image-steganalysis/config_alaska2-image-steganalysis.yaml \
            --task ${DATA_ROOT}/alaska2-image-steganalysis/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/alaska2-image-steganalysis/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "billion-word-imputation")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/billion-word-imputation/config_billion-word-imputation.yaml \
            --task ${DATA_ROOT}/billion-word-imputation/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/billion-word-imputation/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "cassava-leaf-disease-classification")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/cassava-leaf-disease-classification/config_cassava-leaf-disease-classification.yaml \
            --task ${DATA_ROOT}/cassava-leaf-disease-classification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/cassava-leaf-disease-classification/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "cdiscount-image-classification-challenge")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/cdiscount-image-classification-challenge/config_cdiscount-image-classification-challenge.yaml \
            --task ${DATA_ROOT}/cdiscount-image-classification-challenge/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/cdiscount-image-classification-challenge/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "chaii-hindi-and-tamil-question-answering")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/chaii-hindi-and-tamil-question-answering/config_chaii-hindi-and-tamil-question-answering.yaml \
            --task ${DATA_ROOT}/chaii-hindi-and-tamil-question-answering/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/chaii-hindi-and-tamil-question-answering/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "champs-scalar-coupling")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/champs-scalar-coupling/config_champs-scalar-coupling.yaml \
            --task ${DATA_ROOT}/champs-scalar-coupling/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/champs-scalar-coupling/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "facebook-recruiting-iii-keyword-extraction")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/facebook-recruiting-iii-keyword-extraction/config_facebook-recruiting-iii-keyword-extraction.yaml \
            --task ${DATA_ROOT}/facebook-recruiting-iii-keyword-extraction/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/facebook-recruiting-iii-keyword-extraction/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "freesound-audio-tagging-2019")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/freesound-audio-tagging-2019/config_freesound-audio-tagging-2019.yaml \
            --task ${DATA_ROOT}/freesound-audio-tagging-2019/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/freesound-audio-tagging-2019/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "google-quest-challenge")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/google-quest-challenge/config_google-quest-challenge.yaml \
            --task ${DATA_ROOT}/google-quest-challenge/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/google-quest-challenge/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "h-and-m-personalized-fashion-recommendations")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/h-and-m-personalized-fashion-recommendations/config_h-and-m-personalized-fashion-recommendations.yaml \
            --task ${DATA_ROOT}/h-and-m-personalized-fashion-recommendations/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/h-and-m-personalized-fashion-recommendations/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "herbarium-2020-fgvc7")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/herbarium-2020-fgvc7/config_herbarium-2020-fgvc7.yaml \
            --task ${DATA_ROOT}/herbarium-2020-fgvc7/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/herbarium-2020-fgvc7/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "herbarium-2021-fgvc8")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/herbarium-2021-fgvc8/config_herbarium-2021-fgvc8.yaml \
            --task ${DATA_ROOT}/herbarium-2021-fgvc8/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/herbarium-2021-fgvc8/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "herbarium-2022-fgvc9")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/herbarium-2022-fgvc9/config_herbarium-2022-fgvc9.yaml \
            --task ${DATA_ROOT}/herbarium-2022-fgvc9/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/herbarium-2022-fgvc9/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "hotel-id-2021-fgvc8")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/hotel-id-2021-fgvc8/config_hotel-id-2021-fgvc8.yaml \
            --task ${DATA_ROOT}/hotel-id-2021-fgvc8/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/hotel-id-2021-fgvc8/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "hubmap-kidney-segmentation")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/hubmap-kidney-segmentation/config_hubmap-kidney-segmentation.yaml \
            --task ${DATA_ROOT}/hubmap-kidney-segmentation/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/hubmap-kidney-segmentation/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "icecube-neutrinos-in-deep-ice")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/icecube-neutrinos-in-deep-ice/config_icecube-neutrinos-in-deep-ice.yaml \
            --task ${DATA_ROOT}/icecube-neutrinos-in-deep-ice/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/icecube-neutrinos-in-deep-ice/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "imet-2020-fgvc7")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/imet-2020-fgvc7/config_imet-2020-fgvc7.yaml \
            --task ${DATA_ROOT}/imet-2020-fgvc7/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/imet-2020-fgvc7/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "inaturalist-2019-fgvc6")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/inaturalist-2019-fgvc6/config_inaturalist-2019-fgvc6.yaml \
            --task ${DATA_ROOT}/inaturalist-2019-fgvc6/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/inaturalist-2019-fgvc6/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "iwildcam-2020-fgvc7")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/iwildcam-2020-fgvc7/config_iwildcam-2020-fgvc7.yaml \
            --task ${DATA_ROOT}/iwildcam-2020-fgvc7/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/iwildcam-2020-fgvc7/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "jigsaw-unintended-bias-in-toxicity-classification")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/jigsaw-unintended-bias-in-toxicity-classification/config_jigsaw-unintended-bias-in-toxicity-classification.yaml \
            --task ${DATA_ROOT}/jigsaw-unintended-bias-in-toxicity-classification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/jigsaw-unintended-bias-in-toxicity-classification/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "kuzushiji-recognition")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/kuzushiji-recognition/config_kuzushiji-recognition.yaml \
            --task ${DATA_ROOT}/kuzushiji-recognition/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/kuzushiji-recognition/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "learning-agency-lab-automated-essay-scoring-2")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/learning-agency-lab-automated-essay-scoring-2/config_learning-agency-lab-automated-essay-scoring-2.yaml \
            --task ${DATA_ROOT}/learning-agency-lab-automated-essay-scoring-2/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/learning-agency-lab-automated-essay-scoring-2/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "lmsys-chatbot-arena")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/lmsys-chatbot-arena/config_lmsys-chatbot-arena.yaml \
            --task ${DATA_ROOT}/lmsys-chatbot-arena/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/lmsys-chatbot-arena/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "multi-modal-gesture-recognition")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/multi-modal-gesture-recognition/config_multi-modal-gesture-recognition.yaml \
            --task ${DATA_ROOT}/multi-modal-gesture-recognition/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/multi-modal-gesture-recognition/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "osic-pulmonary-fibrosis-progression")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/osic-pulmonary-fibrosis-progression/config_osic-pulmonary-fibrosis-progression.yaml \
            --task ${DATA_ROOT}/osic-pulmonary-fibrosis-progression/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/osic-pulmonary-fibrosis-progression/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "petfinder-pawpularity-score")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/petfinder-pawpularity-score/config_petfinder-pawpularity-score.yaml \
            --task ${DATA_ROOT}/petfinder-pawpularity-score/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/petfinder-pawpularity-score/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "plant-pathology-2021-fgvc8")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/plant-pathology-2021-fgvc8/config_plant-pathology-2021-fgvc8.yaml \
            --task ${DATA_ROOT}/plant-pathology-2021-fgvc8/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/plant-pathology-2021-fgvc8/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "seti-breakthrough-listen")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/seti-breakthrough-listen/config_seti-breakthrough-listen.yaml \
            --task ${DATA_ROOT}/seti-breakthrough-listen/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/seti-breakthrough-listen/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "statoil-iceberg-classifier-challenge")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/statoil-iceberg-classifier-challenge/config_statoil-iceberg-classifier-challenge.yaml \
            --task ${DATA_ROOT}/statoil-iceberg-classifier-challenge/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/statoil-iceberg-classifier-challenge/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "tensorflow-speech-recognition-challenge")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/tensorflow-speech-recognition-challenge/config_tensorflow-speech-recognition-challenge.yaml \
            --task ${DATA_ROOT}/tensorflow-speech-recognition-challenge/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/tensorflow-speech-recognition-challenge/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "tensorflow2-question-answering")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/tensorflow2-question-answering/config_tensorflow2-question-answering.yaml \
            --task ${DATA_ROOT}/tensorflow2-question-answering/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/tensorflow2-question-answering/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "tgs-salt-identification-challenge")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/tgs-salt-identification-challenge/config_tgs-salt-identification-challenge.yaml \
            --task ${DATA_ROOT}/tgs-salt-identification-challenge/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/tgs-salt-identification-challenge/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "tweet-sentiment-extraction")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/tweet-sentiment-extraction/config_tweet-sentiment-extraction.yaml \
            --task ${DATA_ROOT}/tweet-sentiment-extraction/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/tweet-sentiment-extraction/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "us-patent-phrase-to-phrase-matching")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/us-patent-phrase-to-phrase-matching/config_us-patent-phrase-to-phrase-matching.yaml \
            --task ${DATA_ROOT}/us-patent-phrase-to-phrase-matching/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/us-patent-phrase-to-phrase-matching/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "uw-madison-gi-tract-image-segmentation")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/uw-madison-gi-tract-image-segmentation/config_uw-madison-gi-tract-image-segmentation.yaml \
            --task ${DATA_ROOT}/uw-madison-gi-tract-image-segmentation/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/uw-madison-gi-tract-image-segmentation/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "ventilator-pressure-prediction")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/ventilator-pressure-prediction/config_ventilator-pressure-prediction.yaml \
            --task ${DATA_ROOT}/ventilator-pressure-prediction/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/ventilator-pressure-prediction/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "whale-categorization-playground")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/whale-categorization-playground/config_whale-categorization-playground.yaml \
            --task ${DATA_ROOT}/whale-categorization-playground/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/whale-categorization-playground/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "3d-object-detection-for-autonomous-vehicles")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/3d-object-detection-for-autonomous-vehicles/config_3d-object-detection-for-autonomous-vehicles.yaml \
            --task ${DATA_ROOT}/3d-object-detection-for-autonomous-vehicles/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/3d-object-detection-for-autonomous-vehicles/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "bms-molecular-translation")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/bms-molecular-translation/config_bms-molecular-translation.yaml \
            --task ${DATA_ROOT}/bms-molecular-translation/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/bms-molecular-translation/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "google-research-identify-contrails-reduce-global-warming")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/google-research-identify-contrails-reduce-global-warming/config_google-research-identify-contrails-reduce-global-warming.yaml \
            --task ${DATA_ROOT}/google-research-identify-contrails-reduce-global-warming/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/google-research-identify-contrails-reduce-global-warming/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "hms-harmful-brain-activity-classification")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/hms-harmful-brain-activity-classification/config_hms-harmful-brain-activity-classification.yaml \
            --task ${DATA_ROOT}/hms-harmful-brain-activity-classification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/hms-harmful-brain-activity-classification/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "iwildcam-2019-fgvc6")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/iwildcam-2019-fgvc6/config_iwildcam-2019-fgvc6.yaml \
            --task ${DATA_ROOT}/iwildcam-2019-fgvc6/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/iwildcam-2019-fgvc6/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "nfl-player-contact-detection")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/nfl-player-contact-detection/config_nfl-player-contact-detection.yaml \
            --task ${DATA_ROOT}/nfl-player-contact-detection/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/nfl-player-contact-detection/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "predict-volcanic-eruptions-ingv-oe")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/predict-volcanic-eruptions-ingv-oe/config_predict-volcanic-eruptions-ingv-oe.yaml \
            --task ${DATA_ROOT}/predict-volcanic-eruptions-ingv-oe/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/predict-volcanic-eruptions-ingv-oe/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "rsna-2022-cervical-spine-fracture-detection")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/rsna-2022-cervical-spine-fracture-detection/config_rsna-2022-cervical-spine-fracture-detection.yaml \
            --task ${DATA_ROOT}/rsna-2022-cervical-spine-fracture-detection/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/rsna-2022-cervical-spine-fracture-detection/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "rsna-breast-cancer-detection")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/rsna-breast-cancer-detection/config_rsna-breast-cancer-detection.yaml \
            --task ${DATA_ROOT}/rsna-breast-cancer-detection/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/rsna-breast-cancer-detection/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "rsna-miccai-brain-tumor-radiogenomic-classification")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/rsna-miccai-brain-tumor-radiogenomic-classification/config_rsna-miccai-brain-tumor-radiogenomic-classification.yaml \
            --task ${DATA_ROOT}/rsna-miccai-brain-tumor-radiogenomic-classification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/rsna-miccai-brain-tumor-radiogenomic-classification/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "siim-covid19-detection")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/siim-covid19-detection/config_siim-covid19-detection.yaml \
            --task ${DATA_ROOT}/siim-covid19-detection/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/siim-covid19-detection/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "smartphone-decimeter-2022")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/smartphone-decimeter-2022/config_smartphone-decimeter-2022.yaml \
            --task ${DATA_ROOT}/smartphone-decimeter-2022/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/smartphone-decimeter-2022/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "stanford-covid-vaccine")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/stanford-covid-vaccine/config_stanford-covid-vaccine.yaml \
            --task ${DATA_ROOT}/stanford-covid-vaccine/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/stanford-covid-vaccine/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-minimize
        ;;

    "vesuvius-challenge-ink-detection")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/vesuvius-challenge-ink-detection/config_vesuvius-challenge-ink-detection.yaml \
            --task ${DATA_ROOT}/vesuvius-challenge-ink-detection/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/vesuvius-challenge-ink-detection/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;

    "vinbigdata-chest-xray-abnormalities-detection")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/vinbigdata-chest-xray-abnormalities-detection/config_vinbigdata-chest-xray-abnormalities-detection.yaml \
            --task ${DATA_ROOT}/vinbigdata-chest-xray-abnormalities-detection/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_code/data_loader_format/vinbigdata-chest-xray-abnormalities-detection/full_code.py \
            --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
            --test-feedback \
            --force-maximize
        ;;
    *)
        echo "Error, not supported task type '\''$AGENT_TYPE'\'' in MLE-Bench"
        exit 1
        ;;
esac
' _ "$AGENT_TYPE"

status=$?

if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
    echo "[timeout] ablation task ${AGENT_TYPE} timed out after 12h"
    send_feishu "EvoMaster ablation task ${AGENT_TYPE} timed out after 12h"
    exit "$status"
elif [ "$status" -eq 0 ]; then
    echo "[done] ablation task ${AGENT_TYPE} finished successfully"
    send_feishu "EvoMaster ablation task ${AGENT_TYPE} finished successfully"
    exit 0
else
    echo "[failed] ablation task ${AGENT_TYPE} failed with exit code ${status}"
    send_feishu "EvoMaster ablation task ${AGENT_TYPE} failed with exit code ${status}"
    exit "$status"
fi