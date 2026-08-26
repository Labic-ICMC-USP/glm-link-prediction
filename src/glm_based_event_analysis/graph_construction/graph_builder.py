import numpy as np
import pandas as pd
import networkx as nx
import os
import pickle
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import dot_score
from haversine import haversine
from functools import partial
from tqdm import tqdm

def zscore_thresholding(similarities, n_std=2):
    '''
    Calcula um limiar de detecção de outlier baseado em um filtro de z-score.
    Permite a customização do número de desvios padrão (n_std) para controlar a sensibilidade do filtro.
        - n_std mais baixo (ex: 1) -> filtro mais sensível, detectando mais outliers.
        - n_std mais alto (ex: 3) -> filtro menos sensível, detectando menos outliers.
    '''
    
    similarity_threshold = similarities.mean() + n_std * similarities.std()

    return similarity_threshold

def iqr_thresholding(similarities, k=3):
    '''
    Calcula um limiar de detecção de outlier baseado em um filtro de IQR (Interquartile Range).
    Permite a customização do fator de multiplicação (k) para controlar a sensibilidade do filtro.
        - k mais baixo (ex: 1.5) -> filtro mais sensível, detectando mais outliers.
        - k mais alto (ex: 3) -> filtro menos sensível, detectando menos outliers.
    '''

    q1 = np.percentile(similarities, 25)
    q3 = np.percentile(similarities, 75)
    iqr = q3 - q1
    similarity_threshold = q3 + k * iqr

    return similarity_threshold

def percentile_thresholding(similarities, percentile=95):
    '''
    Calcula um limiar de detecção de outlier baseado em um filtro de percentil.
    Permite a customização do percentil para controlar a sensibilidade do filtro.
        - percentil mais baixo (ex: 90) -> filtro mais sensível, detectando mais outliers.
        - percentil mais alto (ex: 99) -> filtro menos sensível, detectando menos outliers.
    '''

    similarity_threshold = np.percentile(similarities, percentile)

    return similarity_threshold

def mad_thresholding(similarities, n=3.5):
    '''
    Calcula um limiar de detecção de outlier baseado em um filtro de MAD (Median Absolute Deviation).
    Permite a customização do fator de multiplicação (n) para controlar a sensibilidade do filtro.
        - n mais baixo (ex: 2) -> filtro mais sensível, detectando mais outliers.
        - n mais alto (ex: 4) -> filtro menos sensível, detectando menos outliers.
    '''

    median = np.median(similarities)
    mad = np.median(np.abs(similarities - median))
    similarity_threshold = median + n * (mad / 0.6745)

    return similarity_threshold

def invert_similarity(distances: np.array):
    '''
    Converte uma matriz de distâncias em uma matriz de similaridades usando a função de inversão simples: similarity = 1 / (1 + distance).
    Essa função limita os valores de similaridade entre 0 e 1, preservando a ordem das similaridades (quanto menor a distância, maior a similaridade).
    '''

    similarity_scores = 1 / (1 + distances)

    return similarity_scores

def exponential_decay_similarity(distances: np.array, alpha=1.0):
    '''
    Converte uma matriz de distâncias em uma matriz de similaridades usando a função de decaimento exponencial: similarity = exp(-alpha * distance).
    Essa função limita os valores de similaridade entre 0 e 1, preservando a ordem das similaridades (quanto menor a distância, maior a similaridade).
    O parâmetro alpha controla a taxa de decaimento da similaridade em relação à distância (alpha mais alto -> decaimento mais rápido).
    '''

    similarity_scores = np.exp(-distances/alpha)

    return similarity_scores

def linear_similarity(distances: np.array, max_distance: float):
    '''
    Converte uma matriz de distâncias em uma matriz de similaridades usando a função de normalização min-max: similarity = 1 - (distance - min_distance) / (max_distance - min_distance).
    Essa função limita os valores de similaridade entre 0 e 1, preservando a ordem das similaridades (quanto menor a distância, maior a similaridade).
    Os parâmetros min_distance e max_distance são usados para normalizar as distâncias, garantindo que a distância mínima corresponda a uma similaridade de 1 e a distância máxima corresponda a uma similaridade de 0.
    '''

    similarity_scores = 1 - distances/max_distance

    return similarity_scores

    

class EventGraphBuilder:

    THRESHOLD_MAP = {
        "zscore": zscore_thresholding,
        "iqr": iqr_thresholding,
        "percentile": percentile_thresholding,
        "mad": mad_thresholding
    }

    DISTANCE_TO_SIMILARITY_MAP = {
        "inverse": invert_similarity,
        "exponential": exponential_decay_similarity,
        "linear": linear_similarity
    }


    def __init__(self, config: dict):
        
        # TODO: num futuro, definir um monte de valor padrao aqui e criar um metodo from_config
        self._validate_config(config)
        
        general_config = config.get("general", {})
        constraints_config = config.get("constraints", {})
        conversion_config = config.get("distance_to_similarity", {})

        # general
        self.similarity_method: str = general_config.get("similarity_method", "multiply")
        self.force_recompute: bool = general_config.get("force_recompute", True)
        self.tmp_files_dir: str = general_config.get("tmp_files_dir", "/tmp")
        self.embedding_model: str = general_config.get("embedding_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.encoding_batch_size: int = general_config.get("encoding_batch_size", 128)
        self.distance_to_similarity_method: str = general_config.get("distance_to_similarity_method", "inverse")

        # constraints
        self.use_non_zero_similarities: bool = constraints_config.get("use_non_zero_similarities", True)
        self.semantic_similarity_range: tuple = tuple(map(float, constraints_config.get("semantic_similarity_range", [0, 1])))
        self.spatial_distance_range: tuple = tuple(map(float, constraints_config.get("spatial_distance_range", [0, np.inf])))
        self.time_days_range: tuple = tuple(map(float, constraints_config.get("time_days_range", [0, np.inf])))
        self.filter_type: str = constraints_config.get("filter_type", "iqr")
        self.threshold: float = constraints_config.get("threshold", 1.5)
        self.max_neighbors: int | None = constraints_config.get("max_neighbors", None) # se None, não há limite no número de vizinhos similares que um nó pode ter
        self.force_directed: bool = constraints_config.get("force_directed", False) # se True, o grafo resultante será direcionado, com arestas apontando do evento mais antigo para o mais novo. 

        # thresold function
        self.threshold_func = self.THRESHOLD_MAP.get(self.filter_type)

        # distance to similarity function
        self.temporal_conversion_func: callable = self.DISTANCE_TO_SIMILARITY_MAP[conversion_config["time"]]
        self.spatial_conversion_func: callable = self.DISTANCE_TO_SIMILARITY_MAP[conversion_config["spatial"]]
        self.temporal_conversion_func: callable = partial(self.temporal_conversion_func, **conversion_config["time_args"])
        self.spatial_conversion_func: callable = partial(self.spatial_conversion_func, **conversion_config["spatial_args"])

        # iniciando outros atributos
        self.what_similarities: np.ndarray | None = None
        self.where_distances: np.ndarray | None = None
        self.when_distances: np.ndarray | None = None
        self.reference_similarities: np.ndarray | None = None
        self.unfiltered_reference_similarities: np.ndarray | None = None

        # checar se o caminho para arquivos temporarios existe, caso contrário criar
        if not os.path.exists(self.tmp_files_dir):
            os.makedirs(self.tmp_files_dir)
    

    def _validate_config(self, config: dict):

        # incrementar no futuro com outras opções de agregação e filtragem
        allowed_methods = ["multiply", "sum", "what", "where", "when"]
        general_configs = config.get("general", {})
        if general_configs.get("similarity_method") not in allowed_methods:
            raise ValueError(f"Invalid similarity_method method. Allowed values are: {allowed_methods}. Got: {general_configs.get('similarity_method')}")
        
        constraints_config = config.get("constraints", {})
        allowed_filter_types = ["zscore", "iqr", "percentile", "mad"]
        if constraints_config.get("filter_type") not in allowed_filter_types:
            raise ValueError(f"Invalid filter type. Allowed values are: {allowed_filter_types}. Got: {constraints_config.get('filter_type')}")
        
        conversion_config = config.get("distance_to_similarity", {})
        allowed_methods = ["inverse", "exponential", "linear"]
        if conversion_config.get("time") not in allowed_methods:
            raise ValueError(f"Invalid distance to similarity method for time. Allowed values are: {allowed_methods}. Got: {conversion_config.get('time')}")
        if conversion_config.get("spatial") not in allowed_methods:
            raise ValueError(f"Invalid distance to similarity method for spatial. Allowed values are: {allowed_methods}. Got: {conversion_config.get('spatial')}")

    def _parse_where_info(self):

        assert self.events_df is not None, "Events dataframe not created yet. Call _create_events_df() first."

        # caso o campo tenha sido gerado como uma lista
        self.events_df["where"] = [where[0] if isinstance(where, list) else where for where in self.events_df["where"].values] 

         # extraindo lat e long para os primeiros paises que forem mencionados em cada evento
        coords = [] # (lat, lon)
        default_coord = (0.0, 0.0) # caso não haja localização, usar coordenada default (ex: coordenada do centro do mapa-mundi)

        where_info = self.events_df["where"].values
        parsed_where_info = [] # armazenando o primeiro pais (ou unkown)

        for curr_where in where_info:
            if "locations" in curr_where.keys():
                locations = curr_where["locations"]
                # caso não haja localizações extraidas
                if len(locations) == 0:
                    coords.append(default_coord)
                    parsed_where_info.append("Unknown")
                # extraindo loc da primeira localização mencionada
                else:
                    first_location = locations[0]
                    # verificando se as coords existem
                    if first_location.get("lat") is None or first_location.get("long") is None:
                        coords.append(default_coord)
                        parsed_where_info.append("Unknown")
                    # se existem, adicionar
                    else:
                        coords.append((first_location.get("lat"), first_location.get("long")))
                        parsed_where_info.append(first_location.get("country", "Unknown"))
            # há casos que o modelo nao gera a lista de coords e gera o pais diretamente.
            elif curr_where.get("lat") is not None and curr_where.get("long") is not None:
                coords.append((curr_where.get("lat"), curr_where.get("long")))
                parsed_where_info.append(curr_where.get("country", "Unknown"))
            # caso não haja localização, usar coordenada default
            else:
                coords.append(default_coord)
                parsed_where_info.append("Unknown")

        # validando extrações
        final_where_info = []
        final_coords = []
        for info, coord in zip(parsed_where_info, coords):
            lat, lon = coord
            if isinstance(lat, (int, float)) and \
               isinstance(lon, (int, float)) and \
               (-90 <= lat <= 90) and \
               (-180 <= lon <= 180):
                final_where_info.append(info)
                final_coords.append(coord)
            else:
                final_where_info.append("Unknown")
                final_coords.append(default_coord)
        
        # atualizando coluna do where com os dados 
        self.events_df["where"] = final_where_info
        # adicionando coluna com coordenadas 
        self.events_df["coords"] = final_coords

        # mostransdo a qtd de unk
        n_unknown = sum([1 for where in final_where_info if where == "Unknown"])
        print(f"- Parsed 'where' info. Found {n_unknown} events with unknown location out of {len(final_where_info)} total events.")

        return self


    def _create_events_df(self):
        # transformando a lista de eventos em um dataframe para facilitar o processamento
        self.events_df = pd.DataFrame(self.raw_events)

        return self

    def _remove_duplicates(self):
        # remoção de duplicatas com base no texto do "what".
        n_before = len(self.events_df)
        self.events_df = self.events_df.drop_duplicates(subset=["what"])
        self.events_df = self.events_df.reset_index(drop=True)
        n_after = len(self.events_df)

        print(f"- Removed duplicates based on 'what' field. {n_before - n_after} duplicates removed.")

        return self
    
    def _remove_invalid_dates(self):
        # remoção de eventos com datas inválidas ou faltantes
        n_before = len(self.events_df)
        self.events_df = self.events_df[pd.to_datetime(self.events_df["when"], errors='coerce').notnull()]
        self.events_df = self.events_df.reset_index(drop=True)
        n_after = len(self.events_df)

        print(f"- Removed events with invalid dates. {n_before - n_after} events removed.")

        return self

    def _create_event_ids(self):
        # criação de identificadores unicos para cada evento
        kw2id = lambda kw: "_".join(aux.strip().replace(" ", "_") for aux in kw.split(",")).lower() # padronizando keywords para usar no id do grafo

        self.events_df["when"] = pd.to_datetime(self.events_df["when"])

        for i, row in self.events_df.iterrows():
            kws = kw2id(row["keywords"])
            event_id = f"{i}_{kws}_{row['when'].date()}"
            self.events_df.at[i, "event_id"] = event_id

        return self

    def _attempt_to_load_matrices(self):
        
        # tentando carregar as matrizes de similaridade mais recentes/distancia do disco, para evitar recomputação
        if not self.force_recompute:
            target_files = {
                "what_similarities": "what_similarities.pkl",
                "where_distances": "where_distance_matrix.pkl",
                "when_distances": "when_distance_matrix.pkl"
            }

            for key, filename in target_files.items():
                file_path = os.path.join(self.tmp_files_dir, filename)
                if os.path.exists(file_path):
                    print(f"Loading existing {key} from disk...")
                    matrix = pickle.load(open(file_path, "rb"))

                    # checar se as dimensões estão compatíveis com o número de eventos atuais
                    if matrix.shape[0] == len(self.events_df) and matrix.shape[1] == len(self.events_df):
                        setattr(self, key, matrix)
                    else:
                        print(f"Dimension mismatch for {key} at {file_path}. Will need to compute.")
                else:
                    print(f"No existing file found for {key} at {file_path}. Will need to compute.")
        
        return self
    
    def _compute_what_similarities(self) -> None:

        model = SentenceTransformer(self.embedding_model, device='cpu')
        what = self.events_df["what"].tolist()
     
        what_embeddings = model.encode(what, show_progress_bar=True, batch_size=self.encoding_batch_size, normalize_embeddings=True)
        self.what_similarities = dot_score(what_embeddings, what_embeddings).cpu().numpy()
        self.what_embeddings = what_embeddings

        # atualizando arquivo de distancias
        what_similarities_path = os.path.join(self.tmp_files_dir, "what_similarities.pkl")
        pickle.dump(self.what_similarities, open(what_similarities_path, "wb"))
        print(f"- Saved what_similarities to {what_similarities_path}")

    
    def _compute_where_distances(self) -> None:

        coords = self.events_df["coords"]
        n = len(coords)
        self.where_distances = np.zeros((n, n), dtype=int) # em km!

        for i in range(n):
            for j in range(i + 1, n):
                distance = haversine(coords[i], coords[j])
                self.where_distances[i][j] = self.where_distances[j][i] = distance
                    
        # serializando matriz de distancias
        where_distance_path = os.path.join(self.tmp_files_dir, "where_distance_matrix.pkl")
        pickle.dump(self.where_distances, open(where_distance_path, "wb"))
        print(f"- Saved where_distance matrix to {where_distance_path}")
    
    def _compute_when_distances(self) -> None:

        when_values = self.events_df["when"].values.astype('datetime64[D]').astype(int)
        when_distance_matrix = np.abs(when_values[:, None] - when_values[None, :])

        self.when_distances = when_distance_matrix

        # salvando distancias
        when_distance_path = os.path.join(self.tmp_files_dir, "when_distance_matrix.pkl")
        pickle.dump(self.when_distances, open(when_distance_path, "wb"))
        print(f"- Saved when_distance matrix to {when_distance_path}")  

    def _get_is_newer_mask(self) -> np.array:

        raw_dates = self.events_df["when"].tolist()
        is_newer_mask = np.zeros((len(self.events_df), len(self.events_df)), dtype=bool)

        for i in range(len(raw_dates)):
            for j in range(i + 1, len(raw_dates)):
                is_newer_mask[i, j] = raw_dates[i] < raw_dates[j]
                is_newer_mask[j, i] = ~is_newer_mask[i, j] # a odem importa

        return is_newer_mask

    def _compute_distance_matrices(self):

        # checar se já existe, caso não -> computar
        if self.what_similarities is None:
            self._compute_what_similarities()
        if self.where_distances is None:
            self._compute_where_distances()
        if self.when_distances is None:
            self._compute_when_distances()
        
        return self
    
    def _convert_distances_to_similarities(self):

        self.temporal_similarity_scores = self.temporal_conversion_func(self.when_distances)
        self.distance_similarity_scores = self.spatial_conversion_func(self.where_distances)

        return self

    
    def _create_reference_similarities(self):
        
        # combinação de modalidades
        if self.similarity_method == "multiply":
            self.reference_similarities = self.what_similarities * self.distance_similarity_scores * self.temporal_similarity_scores
        elif self.similarity_method == "sum":
            self.reference_similarities = (self.what_similarities + self.distance_similarity_scores + self.temporal_similarity_scores) / 3
        # modalidades isoladas para análise
        elif self.similarity_method == "what":
            self.reference_similarities = self.what_similarities.copy()
        elif self.similarity_method == "where":
            self.reference_similarities = self.distance_similarity_scores.copy()
        elif self.similarity_method == "when":    
            self.reference_similarities = self.temporal_similarity_scores.copy()
        else:
            raise ValueError(f"Invalid similarity_method method: {self.similarity_method}")

        return self
    
    
    def _preprocess_reference_similarities(self):

        # pre-processando matriz de similaridades
        allowed_semantic_similarity_range = self.semantic_similarity_range # entre 0 e 1
        allowed_spatial_distance_range = self.spatial_distance_range # em km
        allowed_temporal_distance_range = self.time_days_range # em dias

        # ignorar a diagonal para encontrar o limiar de similaridade
        np.fill_diagonal(self.reference_similarities, 0)

        # salvando as similaridades originais antes de aplicar os filtros, para análise posterior
        self.unfiltered_reference_similarities = self.reference_similarities.copy()

        # ignorando pares u e v que não satisfazem as restrições de distância espacial e temporal 
        n = len(self.events_df)
        for seqid_u in range(n):
            for seqid_v in range(seqid_u + 1, n):
                # filtrando por distancia espacial e temporal
                curr_semantic_similarity = self.what_similarities[seqid_u, seqid_v]
                curr_spatial_distance = self.where_distances[seqid_u, seqid_v]
                curr_temporal_distance = self.when_distances[seqid_u, seqid_v]

                if curr_spatial_distance < allowed_spatial_distance_range[0] or curr_spatial_distance > allowed_spatial_distance_range[1]:
                    self.reference_similarities[seqid_u, seqid_v] = self.reference_similarities[seqid_v, seqid_u]  =  0
                elif curr_temporal_distance < allowed_temporal_distance_range[0] or curr_temporal_distance > allowed_temporal_distance_range[1]:
                    self.reference_similarities[seqid_u, seqid_v] = self.reference_similarities[seqid_v, seqid_u] = 0
                elif curr_semantic_similarity < allowed_semantic_similarity_range[0] or curr_semantic_similarity > allowed_semantic_similarity_range[1]:
                    self.reference_similarities[seqid_u, seqid_v] = self.reference_similarities[seqid_v, seqid_u] = 0

        if self.force_directed:
            is_newer_mask = self._get_is_newer_mask()
            # mantendo apenas similaridades onde u é anterior a v, para garantir que as arestas apontem do mais antigo para o mais novo. Assim, o grafo resultante é direcionado e acíclico
            self.reference_similarities = self.reference_similarities * is_newer_mask
            self.unfiltered_reference_similarities = self.unfiltered_reference_similarities * is_newer_mask

        return self
    
    def _create_edges_from_similarities(self):

        # 1) para cada evento/nó, determinar eventos mais similares e adicionar ao grafo
        G = nx.Graph() if not self.force_directed else nx.DiGraph()
        
        graphid2seqid = {} # para mapear o id do grafo pro id sequencial do dataframe, caso seja necessário depois

        for seq_ed, row in self.events_df.iterrows():

            event_id = row["event_id"]
            graphid2seqid[event_id] = seq_ed


            G.add_node(event_id, 
                       what=row["what"], 
                       where=row["where"], # o primeiro pais foi extraido 
                       when=row["when"], 
                       who=row["who"],
                       how=row["how"],
                       why=row["why"],
            )
            
            # definindo o limiar de similaridade com base na distribuição de similaridades do evento atual
            curr_sims  = self.reference_similarities[seq_ed]

            if self.use_non_zero_similarities:
                non_null_sims = curr_sims[curr_sims > 0]

                # pode acontecer muito raramente de todas as similaridades serem 0
                # deixar o nó sem vizinhos
                if len(non_null_sims) == 0:
                    continue

                similarity_threshold = self.threshold_func(non_null_sims, self.threshold)
            else:
                similarity_threshold = self.threshold_func(curr_sims, self.threshold)

            # encontrar eventos similares -> acima do limiar computado
            threshold_mask = curr_sims >= similarity_threshold
            # ordenando similaridades em ordem decrescente para adicionar os vizinhos mais similares primeiro, caso haja limite de max_neighbors
            sorted_indices = np.argsort(curr_sims)[::-1]

            # filtrando os índices ordenados para manter apenas aqueles que estão acima do limiar de similaridade
            similar_events = sorted_indices[threshold_mask[sorted_indices]]

            # filtrando ou nao o numero de vizinhos maximo
            if self.max_neighbors is not None:
                similar_events = similar_events[:self.max_neighbors]

            for sim_event_seq_id in similar_events:
                sim_event = self.events_df.iloc[sim_event_seq_id]
                sim_event_id = sim_event["event_id"]
                
                # adicionar aresta se ainda não existir
                if not G.has_edge(event_id, sim_event_id):
                    G.add_edge(event_id, sim_event_id, 
                            weight=self.reference_similarities[seq_ed, sim_event_seq_id],
                            what_sim=self.what_similarities[seq_ed, sim_event_seq_id],
                            where_sim=self.distance_similarity_scores[seq_ed, sim_event_seq_id],
                            when_sim=self.temporal_similarity_scores[seq_ed, sim_event_seq_id],
                            edge_type="similarity"
                    )

                    # para ajudar nos próximos passos, se a aresta for adicionada, a similaridade vira 0
                    self.reference_similarities[seq_ed, sim_event_seq_id] = 0
        
        # armazenando produtos desta etapa
        self.graphid2seqid = graphid2seqid
        self.G = G

        print(f"- Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges. Number of connected components: {nx.number_connected_components(G.to_undirected())}")

        return self
    
    def _connect_components(self):

        # 2) conectar eventuais componentes conexas do grafo, para garantir que o grafo final seja conectado, ou seja, haja um caminho entre quaisquer dois eventos.

        # aliases
        G = self.G
        graphid2seqid = self.graphid2seqid
        seqid2graphid = {v: k for k, v in graphid2seqid.items()}

        # funções auxiliares
        connected_components = lambda x: nx.connected_components(x.to_undirected()) if self.force_directed else nx.connected_components
        n_connected_components_func = lambda x: nx.number_connected_components(x.to_undirected()) if self.force_directed else nx.number_connected_components

        to_connect = n_connected_components_func(G)
        # TODO: este código esta muito lento

        # conectar a maior componente conexa com a segunda maior
        with tqdm(total=to_connect - 1, desc="Connecting components") as pbar:
            while n_connected_components_func(G) > 1:
                # lista ordenada de componentes conexas, do maior pro menor           
                cc_list = sorted(connected_components(G), key=len, reverse=True)

                largest_cc = cc_list[0]
                second_largest_cc = cc_list[1]

                largest_cc_nodes = list(largest_cc)
                second_largest_cc_nodes = list(second_largest_cc)

                largest_cc_seq_ids = [graphid2seqid[node] for node in largest_cc_nodes]
                second_largest_cc_seq_ids = [graphid2seqid[node] for node in second_largest_cc_nodes]

                # encontrar o par de eventos (um de cada componente) mais similares
                # TODO: aqui potencialmente estamos violando as restrições de *distancia temporal*
                max_sim = -np.inf
                best_pair = (None, None)
                for seq_id_1 in largest_cc_seq_ids:
                    for seq_id_2 in second_largest_cc_seq_ids:
                        # checar as similaridades originais, antes de aplicar filtros de periodo em dias e distancia espacial
                        sim = self.unfiltered_reference_similarities[seq_id_1, seq_id_2]
                        if sim > max_sim:
                            max_sim = sim
                            best_pair = (seq_id_1, seq_id_2)
                
                # adicionar aresta entre os eventos mais similares das duas componentes
                node_1 = seqid2graphid[best_pair[0]]
                node_2 = seqid2graphid[best_pair[1]]

                if not G.has_edge(node_1, node_2):
                    G.add_edge(node_1, node_2, 
                            weight=max_sim,
                            what_sim=self.what_similarities[best_pair[0], best_pair[1]],
                            where_sim=self.distance_similarity_scores[best_pair[0], best_pair[1]],
                            when_sim=self.temporal_similarity_scores[best_pair[0], best_pair[1]],
                            edge_type="component_bridge"
                    )

                # print(f"- Connected components with edge ({node_1}, {node_2}) with similarity {max_sim}")
                pbar.update(1)
            
        return self
        
        
    def build_graph(self, event_list: list[dict]) -> nx.Graph:

        """Builds an event graph from a list of event dictionaries. The dicts must contain the following keys:
        - "who": list of strings representing the agents involved in the event
        - "what": string description of the event
        - "where": dict with a "locations" key, which is a list of dicts with "lat" and "long" keys.
        - "when": string in the format YYYY-MM-DD representing the date of the event (ISO format)
        - "keywords": string of comma-separated keywords describing the event

        """

        self.raw_events = event_list

        self._create_events_df(). \
             _remove_duplicates(). \
             _remove_invalid_dates(). \
             _parse_where_info(). \
             _create_event_ids(). \
             _attempt_to_load_matrices(). \
             _compute_distance_matrices(). \
             _convert_distances_to_similarities(). \
             _create_reference_similarities(). \
             _preprocess_reference_similarities(). \
             _create_edges_from_similarities(). \
             _connect_components()
        
        return self.G