import pickle
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from functions import *
from sklearn.metrics import silhouette_score
from collections import defaultdict
from tqdm import tqdm

def cluster_field_embeddings(embs, texts, n_clusters):
    k = min(n_clusters, len(embs))
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embs)
    clusters = {i: [] for i in range(k)}
    for text, label in zip(texts, labels):
        clusters[label].append(text)
    centorids = kmeans.cluster_centers_
    representatives = {}
    for cluster_idx in range(k):
        idxs = np.where(labels == cluster_idx)[0]
        if len(idxs) > 0:
            dists = np.linalg.norm(embs[idxs] - centorids[cluster_idx], axis=1)
            closest = idxs[dists.argmin()]
            representatives[cluster_idx] = texts[closest]
        else:
            representatives[cluster_idx] = None
    return clusters, centorids, representatives

def find_optimal_k(embs_array, k_min, k_max):
    ks = list(range(k_min, min(k_max + 1, len(embs_array))))
    inertias = []
    silhouettes = []
    for k in tqdm(ks):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(embs_array)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(embs_array, labels))
    best_k = ks[silhouettes.index(max(silhouettes))]
    return best_k, inertias, silhouettes, ks


def replace_labels_in_data(all_clusters, input_path, fields, output_path, lookup_path):
    with open(input_path, 'r', encoding='utf-8') as rf:
        data = json.load(rf)
    
    lookup = {}
    for field in fields:
        mapping = {}
        clusters = all_clusters.get(field, {}).get('clusters', {})
        representative = all_clusters.get(field, {}).get('representatives', {})
        for cid, labels in clusters.items():
            rep = representative.get(cid)
            for lbl in labels:
                mapping[lbl] = rep
        lookup[field] = mapping
    
    for item in data:
        if "label" in item:
            for field in fields:
                orig = item["label"].get(field)
                if orig and orig in lookup[field]:
                    item["label"][field] = lookup[field][orig]

    with open(lookup_path, 'w', encoding='utf-8') as lf:
        json.dump(lookup, lf, ensure_ascii=False, indent=2)

    formatted = []
    for item in data:
        nested = {}
        if "label" in item:
            nested = {field: item["label"][field] for field in fields if field in item["label"]}
        formatted.append({
            "prompt": item.get("prompt"),
            "label": nested
        })

    with open(output_path, 'w', encoding='utf-8') as wf:
        json.dump(formatted, wf, ensure_ascii=False, indent=2)


def main():
    data_path = 'datasets/SCP/SCP.json'
    pkl_path = 'datasets/SCP/minilm_field_embeddings.pkl'
    clusters_output_path = 'datasets/clusters/final_kmeans/field_clusters_kmeans.json'
    replace_label_path = 'datasets/clusters/final_kmeans/replaced_label_data_kmeans.json'
    lookup_path = 'datasets/clusters/final_kmeans/lookup_table_kmeans.json'
    embeddings = load_pickles(pkl_path)
    all_clusters = {}

    for field, data in tqdm(embeddings.items()):
        embs = np.array(data['embeddings'])
        texts = data['texts']

        if field in ['Role', 'Audience', 'User Intent', 'Tone Type', 'Constraints', 'Reasoning Guidance', 'Output Format']:
            if field == 'Role':
                best_k = 92
            elif field == 'Audience':
                best_k = 47
            elif field == 'User Intent':
                best_k = 15
            elif field == 'Tone Type':
                best_k = 26
            elif field == 'Constraints':
                best_k = 17
            elif field == 'Reasoning Guidance':
                best_k = 10
            elif field == 'Output Format':
                best_k = 15
            # best_k, inertias, silhouettes, ks = find_optimal_k(embs, 10, 51) # 50~1500
            # best_silhouette_score = max(silhouettes)
            # print(f"Field {field} optimal k = {best_k}, silhouette score = {best_silhouette_score}")
            clusters, centroids, representatives = cluster_field_embeddings(embs, texts, best_k)

        elif field == 'Interactive Mode':
            clusters = {
                "0": [
                        "Allowed at the end of response",
                        "Allowed at the end of response for clarification.",
                        "Allowed at the end of response.",
                        "Encourage one follow-up question at the end.",
                        "Encouraged at most one follow-up",
                        "Encouraged at most one follow-up.",
                        "allowed and encouraged to clarify responses or guide the continuation of the interaction, while remaining within the role-play context",
                        "allowed at most one follow-up",
                        "allowed at most one follow-up at the end",
                        "allowed at most one follow-up question at the end",
                        "allowed at the end for clarification",
                        "allowed at the end of each segment to request continuation or clarification",
                        "allowed at the end of response",
                        "allowed at the end of response (to continue code if truncated)",
                        "allowed at the end of response for clarification",
                        "allowed at the end of response for clarification about specific enhancements",
                        "allowed at the end of response for clarification on dataset specifics",
                        "allowed at the end of response for clarification on specific sub-tasks if needed",
                        "allowed at the end of response for clarification or deeper exploration",
                        "allowed at the end of response for clarification or expansion",
                        "allowed at the end of response for clarification or refinement",
                        "allowed at the end of response for clarification or to request more specific resources",
                        "allowed at the end of response for clarifications",
                        "allowed at the end of response for clarifications about specific design elements",
                        "allowed at the end of response for follow-up questions",
                        "allowed at the end of response for further clarification",
                        "allowed at the end of response for further clarification questions",
                        "allowed at the end of response for further questions",
                        "allowed at the end of response if clarification is needed",
                        "allowed at the end of response if clarification is required",
                        "allowed at the end of response if clarification is required to refine interdisciplinary focus",
                        "allowed at the end of response to ask for clarification on scope or focus of the interview",
                        "allowed at the end of response to ask for clarification or additional resources",
                        "allowed at the end of response to clarify any details",
                        "allowed at the end of response to clarify any specific project or contextual details",
                        "allowed at the end of response to clarify any specific technical requirements",
                        "allowed at the end of response to clarify aspects of the passages or analysis",
                        "allowed at the end of response to clarify assumptions or environmental factors",
                        "allowed at the end of response to clarify assumptions or input specifics",
                        "allowed at the end of response to clarify concepts",
                        "allowed at the end of response to clarify criteria or preferences",
                        "allowed at the end of response to clarify data sources or assumptions",
                        "allowed at the end of response to clarify dataset structure",
                        "allowed at the end of response to clarify details",
                        "allowed at the end of response to clarify details or assumptions",
                        "allowed at the end of response to clarify formatting or presentation preferences",
                        "allowed at the end of response to clarify input or gather additional details about user feedback or specific focus areas",
                        "allowed at the end of response to clarify preferences or provide additional data",
                        "allowed at the end of response to clarify project-specific technical elements or requirements",
                        "allowed at the end of response to clarify requirements",
                        "allowed at the end of response to clarify resource details if necessary",
                        "allowed at the end of response to clarify scope or specific topics",
                        "allowed at the end of response to clarify specific aspects of military strategy or sources",
                        "allowed at the end of response to clarify specific aspects or sources",
                        "allowed at the end of response to clarify specific aspects related to the user’s condition",
                        "allowed at the end of response to clarify specific client preferences",
                        "allowed at the end of response to clarify specific concepts or examples",
                        "allowed at the end of response to clarify specific implementation details",
                        "allowed at the end of response to clarify specific needs",
                        "allowed at the end of response to clarify specific sections or requirements",
                        "allowed at the end of response to clarify specific steps if needed",
                        "allowed at the end of response to clarify specific steps or interactive element implementation",
                        "allowed at the end of response to clarify technical details",
                        "allowed at the end of response to clarify technical or conceptual uncertainties",
                        "allowed at the end of response to clarify technical uncertainties",
                        "allowed at the end of response to clarify the dataset or methodology",
                        "allowed at the end of response to clarify the user-specific application",
                        "allowed at the end of response to clarify unclear requirements",
                        "allowed at the end of response to engage with the narrative context",
                        "allowed at the end of response to refine details",
                        "allowed at the end of response to request clarification or additional details",
                        "allowed at the end of response with a follow-up question",
                        "allowed at the end of response'}",
                        "allowed at the end of response, limited to clarifications about the patient’s specific needs or preferences",
                        "allowed at the end of response, limited to clarifying task details",
                        "allowed at the end of response, limited to clarifying the provided factors or goals",
                        "allowed at the end of response, only for clarification of assumptions",
                        "allowed at the end of response, only to clarify data or assumptions",
                        "allowed at the end of response, only to clarify specific formatting or content-related constraints",
                        "allowed at the end of the response",
                        "allowed at the end of the response for clarification",
                        "allowed at the end of the response for clarification on specific methodologies or tools",
                        "allowed at the end of the response for clarification or further guidance",
                        "allowed at the end of the response for clarification questions",
                        "allowed at the end of the response for clarifications",
                        "allowed at the end of the response for clarifying specific steps or details",
                        "allowed at the end of the response for further clarification",
                        "allowed at the end of the response if additional clarification is needed",
                        "allowed at the end of the response to ask for clarification on dataset specifics or preferred visualization tools",
                        "allowed at the end of the response to clarify additional technical details if necessary",
                        "allowed at the end of the response to clarify data points or instructions",
                        "allowed at the end of the response to clarify dataset details or any assumptions",
                        "allowed at the end of the response to clarify details or requirements",
                        "allowed at the end of the response to clarify methodology or reasoning",
                        "allowed at the end of the response to clarify specific aspects of the narrative",
                        "allowed at the end of the response to clarify specific components",
                        "allowed at the end of the response to clarify user intent further if required",
                        "allowed at the end of the response to clarify user preferences",
                        "allowed at the end of the response to confirm understanding or clarify constraints",
                        "allowed at the end of the response to seek clarification or request additional details",
                        "allowed at the end of the response to seek clarification or request deeper focus on specific sub-areas",
                        "allowed at the end, for clarification",
                        "allowed for clarification at the end of the response",
                        "allowed if needed for clarification",
                        "allowed one follow-up question for clarification or additional information",
                        "encouraged after presenting each song's impact",
                        "encouraged at most one follow-up",
                        "encouraged at most one follow-up after delivering full answer",
                        "encouraged at most one follow-up at end",
                        "encouraged at most one follow-up at the end",
                        "encouraged at most one follow-up at the end for clarification",
                        "encouraged at most one follow-up at the end of the response",
                        "encouraged at most one follow-up at the end to confirm specific user preferences",
                        "encouraged at most one follow-up for clarification",
                        "encouraged at most one follow-up for clarification on game mode and opponent's race",
                        "encouraged at most one follow-up for clarification on tools or techniques",
                        "encouraged at most one follow-up for clarification on which story to retell",
                        "encouraged at most one follow-up for clarity",
                        "encouraged at most one follow-up question",
                        "encouraged at most one follow-up question for clarification",
                        "encouraged at most one follow-up question if information is missing",
                        "encouraged at most one follow-up question to indicate press engagement",
                        "encouraged at most one follow-up question, focused on scalability concerns",
                        "encouraged at most one follow-up to clarify device details",
                        "encouraged at most one follow-up to clarify points or engage team members further",
                        "encouraged at most one follow-up to clarify roles or board state",
                        "encouraged at most one follow-up to clarify user's styling goals",
                        "encouraged at most one follow-up to refine or clarify requirements",
                        "encouraged at most one follow-up.",
                        "encouraged at most one follow-up. Few shot Examples: Q: Hi, how can you help me today? A: Hello! I'm here to assist you with any questions or tasks you have. What would you like help with? Q: What's the weather like outside? A: Sure, I can check that for you! Could you let me know your location or the area you're interested in? Q: Can you tell me a joke? A: Of course! Why don't scientists trust atoms? Because they make up everything! Want to hear another one?",
                        "encouraged at multiple points (pose engaging questions to the audience)",
                        "encouraged at the end of response",
                        "encouraged at the end of response for clarification",
                        "encouraged at the end of the response",
                        "encouraged at the start of interaction",
                        "encouraged if necessary to clarify user intent, especially for sensitive topics",
                        "encouraged to ask relevant follow-up questions",
                        "wait for further user input after completing response"
                ],
                "1": [
                        "Not encouraged",
                        "Not encouraged.",
                        "discouraged",
                        "no questions encouraged",
                        "not allowed",
                        "not encouraged",
                        "not encouraged.",
                ]
            }
            representatives = {
                "0": "allowed at the end of response for clarification",
                "1": "not allowed"
            }

        else:
            print(f'처리하지 않는 필드: {field}')
            # fallback으로 k=10 사용
            clusters, centroids, representatives = cluster_field_embeddings(embs, texts, 10)

        all_clusters[field] = {
            'clusters': clusters,
            'representatives': representatives
        }

    with open(clusters_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_clusters, f, ensure_ascii=False, indent=2)

    fields_to_replace = ['Role', 'Audience', 'User Intent', 'Tone Type', 'Constraints', 'Reasoning Guidance', 'Output Format', 'Interactive Mode']
    replace_labels_in_data(
        all_clusters,
        data_path,
        fields_to_replace,
        replace_label_path,
        lookup_path
    )

if __name__ == '__main__':
    main()
