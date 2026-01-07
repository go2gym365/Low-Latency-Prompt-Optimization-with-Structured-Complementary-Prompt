import pickle
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from functions import *
from sklearn.metrics import silhouette_score
from collections import defaultdict
from tqdm import tqdm

def cluster_field_embeddings(embs, texts, n_clusters):
    """AgglomerativeClustering을 사용하여 필드 임베딩을 클러스터링"""
    k = min(n_clusters, len(embs))
    agglomerative = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = agglomerative.fit_predict(embs)
    
    clusters = {i: [] for i in range(k)}
    for text, label in zip(texts, labels):
        clusters[label].append(text)
    
    # AgglomerativeClustering은 centroids가 없으므로 medoid 방식으로 대표값 찾기
    representatives = {}
    for cluster_idx in range(k):
        idxs = np.where(labels == cluster_idx)[0]
        if len(idxs) > 0:
            # 클러스터 내 모든 점들의 평균을 계산
            cluster_embeddings = embs[idxs]
            cluster_center = np.mean(cluster_embeddings, axis=0)
            
            # 평균점에 가장 가까운 실제 데이터 포인트 찾기 (medoid)
            dists = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
            closest = idxs[dists.argmin()]
            representatives[cluster_idx] = texts[closest]
        else:
            representatives[cluster_idx] = None
    
    return clusters, None, representatives  # centroids는 None으로 반환

def find_optimal_n_clusters(embs_array, n_min, n_max):
    """최적의 n_clusters를 실루엣 스코어 기준으로 찾기"""
    ns = list(range(n_min, n_max + 1))
    
    silhouettes = []
    valid_ns = []
    
    print(f"n_clusters 범위: {n_min} ~ {n_max}, 테스트할 값들: {ns}")
    
    for n in tqdm(ns, desc="AgglomerativeClustering 최적화"):
        try:
            agglomerative = AgglomerativeClustering(n_clusters=n, linkage='ward')
            labels = agglomerative.fit_predict(embs_array)
            
            # 실루엣 스코어 계산 (유효한 경우만)
            if n >= 2 and len(set(labels)) > 1:
                sil_score = silhouette_score(embs_array, labels)
                silhouettes.append(sil_score)
                valid_ns.append(n)
        
        except Exception as e:
            print(f"  n_clusters={n}에서 오류 발생: {e}")
    

    
    # 최고 실루엣 스코어의 n_clusters 찾기
    best_idx = silhouettes.index(max(silhouettes))
    best_n = valid_ns[best_idx]
    
    return best_n, [], silhouettes, valid_ns

def replace_labels_in_data(all_clusters, input_path, fields, output_path, lookup_path):
    """원본 데이터의 라벨을 클러스터 대표값으로 교체"""
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
    clusters_output_path = 'datasets/clusters/final_agglomerative/field_clusters_agglomerative.json'
    replace_label_path = 'datasets/clusters/final_agglomerative/replaced_label_data_agglomerative.json'
    lookup_path = 'datasets/clusters/final_agglomerative/lookup_table_agglomerative.json'
    embeddings = load_pickles(pkl_path)
    all_clusters = {}

    for field, data in tqdm(embeddings.items(), desc="필드별 클러스터링"):
        embs = np.array(data['embeddings'])
        texts = data['texts']

        if field in ['Role', 'Audience', 'User Intent', 'Tone Type', 'Constraints', 'Reasoning Guidance', 'Output Format']:
            if field == 'Role':
                best_n = 75
            elif field == 'Audience':
                best_n = 50
            elif field == 'User Intent':
                best_n = 27
            elif field == 'Tone Type':
                best_k = 50
            elif field == 'Constraints':
                best_k = 17
            elif field == 'Reasoning Guidance':
                best_k = 10
            elif field == 'Output Format':
                best_n = 14
            # print(f"\n{field} 필드 AgglomerativeClustering 최적화 중... (데이터 수: {len(embs)})")
            # best_n, _, silhouettes, ns = find_optimal_n_clusters(embs, 50, 100)  # 50~500 범위
            
            # 유효한 실루엣 점수만 사용
            # valid_silhouettes = [s for s in silhouettes if s > -1]
            # best_silhouette_score = max(valid_silhouettes) if valid_silhouettes else -1
            
            # print(f"Field {field} optimal n_clusters = {best_n}, silhouette score = {best_silhouette_score}")

            clusters, _, representatives = cluster_field_embeddings(embs, texts, best_n)

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
            # fallback으로 n_clusters=10 사용
            clusters, _, representatives = cluster_field_embeddings(embs, texts, 10)

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