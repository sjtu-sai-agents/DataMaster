#!/bin/bash

AGENT_TYPE=$1

if [ -z "$AGENT_TYPE" ]; then
    echo "Error: please input agent type"
    exit 1
fi


case "$AGENT_TYPE" in
    "dogs-vs-cats-redux-kernels-edition_resnet50_basics")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/dogs-vs-cats-redux-kernels-edition/config_dogs-vs-cats-redux-kernels-edition_resnet50_basics.yaml \
            --task ${DATA_ROOT}/dogs-vs-cats-redux-kernels-edition/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_codebase/dogs-vs-cats-redux-kernels-edition/solution_00106d7f3f6940de822bc47a99bca89a.py \
        ;;

    "dog-breed-identification_resnet50_basics")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/dog-breed-identification/config_dog-breed-identification_resnet50_basics.yaml \
            --task ${DATA_ROOT}/dog-breed-identification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_codebase/dog-breed-identification/solution_96bc46799a5e4a43a0c6cefee332e17e.py \
        ;;

    "jigsaw-toxic-comment-classification-challenge_transformer_basic")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/jigsaw-toxic-comment-classification-challenge/config_jigsaw-toxic-comment-classification-challenge_transformer_basic.yaml \
            --task ${DATA_ROOT}/jigsaw-toxic-comment-classification-challenge/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_codebase/jigsaw-toxic-comment-classification-challenge/solution_cc7ad038d3944e048c4f4c13e011d3e9.py \
        ;;

    "random-acts-of-pizza_BERT")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/random-acts-of-pizza/config_random-acts-of-pizza_BERT.yaml \
            --task ${DATA_ROOT}/random-acts-of-pizza/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_codebase/random-acts-of-pizza/solution_0be7d3f5b6dc4565a19303b5f6164e91.py \
        ;;

    "the-icml-2013-whale-challenge-right-whale-redux_v1")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/the-icml-2013-whale-challenge-right-whale-redux/config_the-icml-2013-whale-challenge-right-whale-redux_v1.yaml \
            --task ${DATA_ROOT}/the-icml-2013-whale-challenge-right-whale-redux/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_codebase/the-icml-2013-whale-challenge-right-whale-redux/solution_cabfd2cafce64ecabacb12dffee10d12.py \
        ;;

    "text-normalization-challenge-russian-language_v1")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/text-normalization-challenge-russian-language/config_text-normalization-challenge-russian-language_v1.yaml \
            --task ${DATA_ROOT}/text-normalization-challenge-russian-language/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_codebase/text-normalization-challenge-russian-language/solution_65a88e47ada241fd930343123a8532ee.py \
        ;;

    "siim-isic-melanoma-classification_v1")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/siim-isic-melanoma-classification/config_siim-isic-melanoma-classification_v1.yaml \
            --task ${DATA_ROOT}/siim-isic-melanoma-classification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_codebase/siim-isic-melanoma-classification/solution_b7027affe35f4f8ab2963615712b1a87.py \
        ;;

    "dog-breed-identification_effnetb3")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/dog-breed-identification/config_dog-breed-identification_effnetb3.yaml \
            --task ${DATA_ROOT}/dog-breed-identification/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_codebase/dog-breed-identification/solution_578f8e5cab9348e0a1c7468f18d1b13e.py \
        ;;

    "histopathologic-cancer-detection_resnet34")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/histopathologic-cancer-detection/config_histopathologic-cancer-detection_resnet34.yaml \
            --task ${DATA_ROOT}/histopathologic-cancer-detection/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_codebase/histopathologic-cancer-detection/solution_08a9cb4cad9b4cd4819306ebe37f511b.py \
        ;;

    "nomad2018-predict-transparent-conductors_dl")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/nomad2018-predict-transparent-conductors/config_nomad2018-predict-transparent-conductors_dl.yaml \
            --task ${DATA_ROOT}/nomad2018-predict-transparent-conductors/description.md \
            --initial-code ${PROJECT_ROOT}/initial_codebase/nomad2018-predict-transparent-conductors/solution_6b628c5a7c6a473cb3e79830794b4210.py \
        ;;

    "text-normalization-challenge-english-language_t5_1")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/text-normalization-challenge-english-language/config_text-normalization-challenge-english-language_t5_1.yaml \
            --task ${DATA_ROOT}/text-normalization-challenge-english-language/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_codebase/text-normalization-challenge-english-language/solution_232bbf74e1b14eb284db956f115a0119.py \
        ;;

    "tabular-playground-series-dec-2021_v2")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/tabular-playground-series-dec-2021/config_tabular-playground-series-dec-2021_v2.yaml \
            --task ${DATA_ROOT}/tabular-playground-series-dec-2021/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_codebase/tabular-playground-series-dec-2021/solution_457b73633d0543bba0026bad33130f1a.py \
        ;;

    "tabular-playground-series-may-2022_v2")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/yaml_configs/tabular-playground-series-may-2022/config_tabular-playground-series-may-2022_v2.yaml \
            --task ${DATA_ROOT}/tabular-playground-series-may-2022/prepared/public/description.md \
            --initial-code ${PROJECT_ROOT}/initial_codebase/tabular-playground-series-may-2022/solution_00820cb3b6f24167a5108c48ccd14d5d.py \
        ;;

    *)
        echo "Error, not supported task type '$AGENT_TYPE' in MLE-Bench"
        exit 1
        ;;
esac