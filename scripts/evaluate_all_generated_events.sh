OUTPUT_DIR=../data/generated/temp1.2
LP_FT_MODEL=/exp_local/kenzosaki/models/ai_events_gemma_3_4b_ft/checkpoint-680
# echo "Ollama"
# python3 link_prediction/evaluate_links.py --generated_events $OUTPUT_DIR/ollama_no_neigh.json --model $LP_FT_MODEL
# python3 link_prediction/evaluate_links.py --generated_events $OUTPUT_DIR/ollama_neigh.json --model $LP_FT_MODEL

echo "CPT"
#python3 link_prediction/evaluate_links.py --generated_events $OUTPUT_DIR/cpt_no_neigh.json --model $LP_FT_MODEL
python3 link_prediction/evaluate_links.py --generated_events $OUTPUT_DIR/cpt_neigh.json --model $LP_FT_MODEL

echo "DPO"
#python3 link_prediction/evaluate_links.py --generated_events $OUTPUT_DIR/dpo_no_neigh.json --model $LP_FT_MODEL
python3 link_prediction/evaluate_links.py --generated_events $OUTPUT_DIR/dpo_neigh.json --model $LP_FT_MODEL