OUTPUT_DIR=../data/generated/temp1.2
GRAPH_PATH=../data/processed/directed_mad3.5_ai_news_event_graph.pkl
SOURCE_NODES=../data/processed/expansion_edges.json
OLLAMA_CONFIG=../src/glm_based_event_analysis/config/generation/ollama.yaml
CPT_CONFIG=../src/glm_based_event_analysis/config/generation/cpt.yaml
DPO_CONFIG=../src/glm_based_event_analysis/config/generation/dpo.yaml

echo "Generating events with Ollama"
#python generation/generate_events.py --generation_config $OLLAMA_CONFIG \
#                          --graph $GRAPH_PATH \
#                          --source_nodes $SOURCE_NODES \
#                          --output_path $OUTPUT_DIR/ollama_no_neigh.json

#python generation/generate_events.py --generation_config $OLLAMA_CONFIG \
#                          --graph $GRAPH_PATH \
#                          --source_nodes $SOURCE_NODES \
#                          --output_path $OUTPUT_DIR/ollama_neigh.json \
#                          --use_neighbours

echo "Generating events with CPT"        
# python generation/generate_events.py --generation_config $CPT_CONFIG \
#                           --graph $GRAPH_PATH \
#                           --source_nodes $SOURCE_NODES \
#                           --output_path $OUTPUT_DIR/cpt_no_neigh.json

python generation/generate_events.py --generation_config $CPT_CONFIG \
                          --graph $GRAPH_PATH \
                          --source_nodes $SOURCE_NODES \
                          --output_path $OUTPUT_DIR/cpt_neigh.json \
                          --use_neighbours

echo "Generating events with DPO"
# python generation/generate_events.py --generation_config $DPO_CONFIG \
#                          --graph $GRAPH_PATH \
#                          --source_nodes $SOURCE_NODES \
#                          --output_path $OUTPUT_DIR/dpo_no_neigh.json

python generation/generate_events.py --generation_config $DPO_CONFIG \
                         --graph $GRAPH_PATH \
                         --source_nodes $SOURCE_NODES \
                         --output_path $OUTPUT_DIR/dpo_neigh.json \
                         --use_neighbours