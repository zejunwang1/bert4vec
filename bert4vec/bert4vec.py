#!usr/bin/env python
# -*- coding:utf-8 -*-

import os
import math
import logging
import numpy as np
from transformers import (
    BertTokenizerFast, 
    BertModel, 
    RoFormerModel, 
    AutoTokenizer, 
    AutoModel
)
from typing import Union, List
from numpy import ndarray
import torch
import torch.nn as nn

from . import __version__
from .util import snapshot_download

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class RoFormerModelWithPooler(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.roformer = RoFormerModel.from_pretrained(model_path)
        
        model_file = os.path.join(model_path, "pytorch_model.bin")
        assert os.path.isfile(model_file)
        params_dict = torch.load(model_file)
        try:
            pooler_weight = params_dict["pooler.dense.weight"]
            pooler_bias = params_dict["pooler.dense.bias"]
        except:
            # model with new conversion script convert_roformer_sim_original_tf_checkpoint_to_pytorch.py
            pooler_weight = params_dict["roformer.pooler.weight"]
            pooler_bias = params_dict["roformer.pooler.bias"]
        del params_dict
        self.pooler = nn.Linear(pooler_weight.shape[0], pooler_weight.shape[0])
        self.pooler.weight.data = pooler_weight
        self.pooler.bias.data = pooler_bias
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None
    ): 
        outputs = self.roformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)
        
        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]
        pooled_output = self.pooler(cls_output)
        return (sequence_output, pooled_output) + outputs[1:]

class Bert4Vec(object):
    def __init__(self,
                 mode: str = 'roformer-sim-small',
                 model_name_or_path: str = 'WangZeJun/roformer-sim-small-chinese'):
        """
        Args:
            mode: str, "simbert-base" 使用simbert-base-chinese模型获得句向量
                       "roformer-sim-base" 使用roformer-sim-base-chinese模型获得句向量
                       "roformer-sim-small" 使用roformer-sim-small-chinese模型获得句向量
                       "paraphrase-multilingual-minilm" 使用sentence-transformers的paraphrase-multilingual-MiniLM-L12-v2模型获得句向量
            model_name_or_path: str, 句向量生成模型名称或存储路径, 模型名称与mode的对应关系为
                                     mode="simbert-base", model_name_or_path="WangZeJun/simbert-base-chinese"
                                     mode="roformer-sim-base", model_name_or_path="WangZeJun/roformer-sim-base-chinese"
                                     mode="roformer-sim-small", model_name_or_path="WangZeJun/roformer-sim-small-chinese"
                                     mode="paraphrase-multilingual-minilm", model_name_or_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert mode in ["simbert-base", "roformer-sim-base", "roformer-sim-small", "roformer-sim-ft-small", "roformer-sim-ft-base", "paraphrase-multilingual-minilm"]
        self.mode = mode
        if mode == "simbert-base":
            if not os.path.isdir(model_name_or_path):
                model_name_or_path = "WangZeJun/simbert-base-chinese"
            self.tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
            self.model = BertModel.from_pretrained(model_name_or_path)
        elif "roformer" in mode:
            if not os.path.isdir(model_name_or_path):
                if mode == "roformer-sim-base":
                    model_name_or_path = "WangZeJun/roformer-sim-base-chinese"
                elif mode == "roformer-sim-ft-small":
                    model_name_or_path = "blmoistawinde/roformer-sim-ft-small-chinese"
                elif mode == "roformer-sim-ft-base":
                    model_name_or_path = "blmoistawinde/roformer-sim-ft-base-chinese"
                else:
                    model_name_or_path = "WangZeJun/roformer-sim-small-chinese"
                try:
                    from torch.hub import _get_torch_home
                    torch_cache_home = _get_torch_home()
                except ImportError:
                    torch_cache_home = os.path.expanduser(os.getenv('TORCH_HOME', os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
                
                cache_folder = os.path.join(torch_cache_home, 'bert4vec')
                model_path = os.path.join(cache_folder, model_name_or_path.replace("/", "_"))
                if not os.path.exists(model_path):
                    model_path = snapshot_download(model_name_or_path,
                                                   cache_dir=cache_folder,
                                                   library_name='bert4vec',
                                                   library_version=__version__)
            else:
                model_path = model_name_or_path

            self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
            self.model = RoFormerModelWithPooler(model_path)
        else:
            if not os.path.isdir(model_name_or_path):
                model_name_or_path = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
            self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.eval()
        self.model.to(self.device)

    def mean_pooling(self, token_embeddings, attention_mask):
        attention_mask = torch.unsqueeze(attention_mask, dim=-1)
        token_embeddings = token_embeddings * attention_mask
        seqlen = torch.sum(attention_mask, dim=1)
        embeddings = torch.sum(token_embeddings, dim=1) / seqlen
        return embeddings

    def encode(self,
               sentences: Union[str, List[str]],
               batch_size: int = 64,
               convert_to_numpy: bool = False,
               normalize_to_unit: bool = False):
        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        if len(sentences) < 1:
            return all_embeddings
        length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        num_batches = int((len(sentences) - 1) / batch_size) + 1
        with torch.no_grad():
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(sentences_sorted))
                inputs = self.tokenizer(
                    sentences_sorted[start:end],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**inputs)
                if self.mode in ["simbert-base", "roformer-sim-base", "roformer-sim-small", "roformer-sim-ft-base", "roformer-sim-ft-small"]:
                    embeddings = outputs[1]
                else:
                    embeddings = self.mean_pooling(outputs[0], inputs["attention_mask"])
                if normalize_to_unit:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                if convert_to_numpy:
                    embeddings = embeddings.cpu()
                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        else:
            all_embeddings = torch.stack(all_embeddings)
        if input_was_string:
            all_embeddings = all_embeddings[0]
        return all_embeddings

    def similarity(self,
                   queries: Union[str, List[str]],
                   keys: Union[str, List[str], ndarray],
                   batch_size: int = 64,
                   return_matrix: bool = False):
        query_vecs = self.encode(queries, batch_size=batch_size, 
            normalize_to_unit=True)
        
        if not isinstance(keys, ndarray):
            key_vecs = self.encode(keys, batch_size=batch_size,
                normalize_to_unit=True)
        else:
            key_vecs = keys

        single_query, single_key = len(query_vecs.shape) == 1, len(key_vecs.shape) == 1
        if single_query:
            query_vecs = query_vecs.unsqueeze(0)
        if single_key:
            if isinstance(key_vecs, ndarray):
                key_vecs = key_vecs.reshape(1, -1)
            else:
                key_vecs = key_vecs.unsqueeze(0)
        
        if return_matrix:
            if isinstance(key_vecs, ndarray):
                query_vecs = query_vecs.cpu().numpy()
                similarity = np.matmul(query_vecs, np.transpose(key_vecs))
            else:
                similarity = torch.mm(query_vecs, key_vecs.transpose(0, 1)).cpu().numpy()
            if single_query:
                similarity = similarity[0]
                if single_key:
                    similarity = float(similarity[0])
        else:
            assert query_vecs.shape[0] == key_vecs.shape[0]
            if isinstance(key_vecs, ndarray):
                query_vecs = query_vecs.cpu().numpy()
                similarity = np.sum(query_vecs * key_vecs, axis=-1)
            else:
                similarity = torch.sum(query_vecs * key_vecs, dim=-1).cpu().numpy()
            if single_query:
                similarity = float(similarity[0])
        return similarity
            
    def build_index(self,
                    sentences_or_file_path: Union[str, List[str]],
                    ann_search: bool = False,
                    gpu_index: bool = False,
                    gpu_memory: int = 16,
                    n_search: int = 64,
                    batch_size: int = 64):
        try:
            import faiss
            assert hasattr(faiss, "IndexFlatIP")
            use_faiss = True
        except:
            logger.warning("Fail to import faiss. If you want to use faiss, install faiss through PyPI. Now the program continues with brute force search.")
            use_faiss = False

        if isinstance(sentences_or_file_path, str):
            sentences = []
            with open(sentences_or_file_path, "r", encoding="utf-8") as f:
                logging.info("Loading sentences from %s" % (sentences_or_file_path))
                for line in f:
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences
        
        logger.info("Encoding embeddings for sentences")
        embeddings = self.encode(sentences_or_file_path, batch_size=batch_size, normalize_to_unit=True, convert_to_numpy=True)
        
        logger.info("Building index")
        self.index = {"sentences": sentences_or_file_path}
        if use_faiss:
            d = embeddings.shape[1]
            nlist = int(math.sqrt(embeddings.shape[0]))
            quantizer = faiss.IndexFlatIP(d)
            if ann_search:
                index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                index = quantizer 
            
            if gpu_index:
                if hasattr(faiss, "StandardGpuResources"):
                    logger.info("Use GPU-version faiss")
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(gpu_memory * 1024 * 1024 * 1024)
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                else:
                    logger.info("Use CPU-version faiss")
            
            if ann_search:
                index.train(embeddings.astype(np.float32))
            index.add(embeddings.astype(np.float32))
            index.nprobe = min(nlist, n_search)
            self.is_faiss_index = True
        else:
            index = embeddings
            self.is_faiss_index = False
        self.index["index"] = index
        
    def search(self,
               queries: Union[str, List[str]],
               threshold: float = 0.0,
               top_k: int = 5):
        if not self.is_faiss_index:
            if isinstance(queries, list):
                all_results = []
                for query in queries:
                    results = self.search(
                        queries=query,
                        threshold=threshold,
                        top_k=top_k)
                    all_results.append(results)
                return all_results
            similarity = self.similarity(queries, self.index["index"], return_matrix=True).tolist()
            id_scores = []
            for i, s in enumerate(similarity):
                if s >= threshold:
                    id_scores.append((i, s))
            id_scores = sorted(id_scores, key=lambda x: x[1], reverse=True)[:top_k]
            results = [(self.index["sentences"][idx], score) for idx, score in id_scores]
            return results
        else:
            query_vecs = self.encode(queries, normalize_to_unit=True, convert_to_numpy=True)
            if isinstance(queries, str):
                query_vecs = np.expand_dims(query_vecs, axis=0)
            
            distance, idx = self.index["index"].search(query_vecs.astype(np.float32), top_k)
            
            def single_result(dist, idx):
                results = [(self.index["sentences"][i], s) for i, s in zip(idx, dist) if s >= threshold]
                return results
            
            if isinstance(queries, list):
                all_results = []
                for i in range(len(queries)):
                    results = single_result(distance[i], idx[i])
                    all_results.append(results)
                return all_results
            return single_result(distance[0], idx[0])

    def write_index(self, index_path: str):
        if self.is_faiss_index:
            import faiss
            faiss.write_index(self.index["index"], index_path)
        else:
            np.savez(index_path, index=self.index["index"])
    
    def read_index(self, sentences_path: str, index_path: str, is_faiss_index: bool = True):
        assert os.path.isfile(sentences_path)
        assert os.path.isfile(index_path)
        self.is_faiss_index = is_faiss_index
        if is_faiss_index:
            import faiss
            index = faiss.read_index(index_path)
        else:
            index = np.load(index_path)["index"]
        sentences = []
        with open(sentences_path, "r", encoding="utf-8") as f:
            for line in f:
                sentences.append(line.rstrip())
        self.index = {"sentences": sentences}
        self.index["index"] = index


