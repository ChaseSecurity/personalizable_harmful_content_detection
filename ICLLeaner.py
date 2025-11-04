
import pandas as pd
# from retriv import SparseRetriever, DenseRetriever
from vllm import LLM, SamplingParams
from typing import List
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import json
from typing import List, Literal
from typing import Dict
import os
import warnings
import seaborn as sns
from vllm import TokensPrompt
from vllm.sampling_params import GuidedDecodingParams

class InContextLearner:
    def __init__(
            self,
            task_type: str, # 'specific_binary', 'binary', 'multi-class-3-1', 'multi-class-3-3', 'multi-label'
            reasoning: bool, # whether to enable the reasoning in the prompt
            model_name: str, 
            train_df: pd.DataFrame, # training data
            test_df: pd.DataFrame, # testing data
            context_length: int, # context length of the model 
            model: LLM = None, # if not None, use the provided model, otherwise load the model from model_name
            base_dir: str = "/data/rfzhang/harmfultext/model/", # base directory for the model
            index_name = "training-examples", # index name for the retriever
    ) -> None:
        # Initialize parameters
        self.train_df = train_df
        self.test_df = test_df
        self.model = model
        self.model_context_length = context_length
        self.index_name = index_name   # index name for the retriever
        self.tokenizer = self.model.get_tokenizer()
        self.train_df_list = []
        self.retriever_list = []
        self.retriever = None
        self.task_type = task_type
        self.reasoning = reasoning
        if self.model == None:
            model_dir = base_dir + model_name.split('/')[1]
            self.model = LLM(model=model_dir, trust_remote_code=True)
            self.model_context_length = context_length

    def create_retriever(
            self,
            retrieval: str, # 'lexical', 'semantic'
            index_name: str, # index name for the retriever
            df: pd.DataFrame, # training data
        ):
            # Prepare the collection of documents for indexing
            collection = [{"id": idx, "text": row["text"]} for idx, row in df.iterrows()]
            # Initialize the appropriate retriever based on the retrieval method
            if retrieval == 'lexical': # https://github.com/AmenRa/retriv/blob/main/docs/sparse_retriever.md
                retriever = SparseRetriever(
                    index_name=index_name,
                    model="bm25",
                    min_df=1, # represent the minimum frequency of occurrence of terms in the document set
                    tokenizer="whitespace",
                    stemmer='english',  # Not support Chinese https://github.com/AmenRa/retriv/blob/main/docs/text_preprocessing.md
                    stopwords='english', # Support only single language
                    do_lowercasing=True,
                    do_ampersand_normalization=True,        # & -> and
                    do_special_chars_normalization=False,   # e.g. übermensch → ubermensch
                    do_acronyms_normalization=False,        # e.g. U.S.A. -> USA
                    do_punctuation_removal=False,
                ).index(collection)

            elif retrieval == 'semantic': # https://github.com/AmenRa/retriv/blob/main/docs/dense_retriever.md
                retriever = DenseRetriever(
                    index_name=index_name,
                    model="/data/rfzhang/harmfultext/model/sentence-transformers/",
                    normalize=True, 
                    max_length=128,
                    use_ann=False,
                ).index(collection, use_gpu=True)
            #else if retrieval = 'hybrid': # https://github.com/AmenRa/retriv/blob/main/docs/hybrid_retriever.md
            else:
                assert False, f"Retrieval method {retrieval} is not supported"

            return retriever
    def generate_prompts(
            self,
            system_description: str,
            n_shots: int = 0, # the sum number of shots in the prompt
            retrieval: Literal['random', 'lexical', 'semantic', 'balanced semantic', 'balanced lexical', 'Fine-grained balanced semantic', 'Fine-grained balanced lexical'] = 'random', # For single-task ICL, you can use 'random' or 'lexical' or'semantic' or 'balanced semantic' or 'balanced lexical', for multi-task ICL, you can use 'random' or 'lexical' or'semantic' or 'balanced semantic' or 'balanced lexical' orFine-grained balanced semantic' or 'Fine-grained balanced lexical'
            random_seed: int = 42,
    ):     
        prompts = [] # List of TokensPrompt
        labels = self.train_df['label'].unique()
        # Initialize the retriever based on the retrieval method if not random
        if retrieval != 'random' and n_shots > 0 and self.retriever is None and len(self.retriever_list) == 0:
            if 'balanced' in retrieval:
                if 'Fine-grained' in retrieval:
                    original_labels = self.train_df['original_label'].unique()
                    for original_label in original_labels:
                        temp_df = self.train_df[self.train_df['original_label'] == original_label]
                        temp_df.reset_index(inplace=True, drop=True)
                        self.train_df_list.append(temp_df)
                        self.retriever_list.append(self.create_retriever(retrieval.split(' ')[-1], retrieval+'_' + str(len(self.retriever_list)), temp_df))
                else:
                    for label in labels:
                        temp_df = self.train_df[self.train_df['label'] == label]
                        temp_df.reset_index(inplace=True, drop=True)
                        self.train_df_list.append(temp_df)
                        self.retriever_list.append(self.create_retriever(retrieval.split(' ')[-1], retrieval+'_' + str(len(self.retriever_list)), temp_df))
            else:
                retriever = self.create_retriever(retrieval, retrieval, self.train_df)
        # get the number of texts belonging to each label in the prompt to check the correctness of the prompt
        # category_count = {}
        # for original_label in self.train_df['label'].unique():
        #     category_count[original_label] = {}
        #     for original_label2 in self.train_df['original_label'].unique():
        #         category_count[original_label][original_label2] = 0
            
        for query, label in zip(self.test_df['text'], self.test_df['label']):
            # If retrieval method is random, sample from training data
            # demos = demos_df[demos_df['text'] == query].iloc[0][demos_column_name]
            if retrieval == 'random' or n_shots == 0:
                sampled_demos = self.train_df.sample(n_shots, random_state=random_seed)
            elif 'balanced' in retrieval:
                sampled_demos = None
                for i, retriever in enumerate(self.retriever_list):
                    retrieved_data = retriever.search(
                        query=query,
                        cutoff=int(n_shots/len(self.retriever_list)),
                    )
                    ids = [item['id'] for item in retrieved_data]
                    # If not enough examples are retrieved, sample randomly to fill the gap
                    if len(ids) < int(n_shots/len(self.retriever_list)):
                            warnings.warn(f"Not enough examples retrieved for query {query}, adding more examples by sampling from the remaining examples")
                            remaining_examples = self.train_df_list[i].drop(index=ids)
                            ids.extend(remaining_examples.sample(int(n_shots/len(self.retriever_list))-len(ids), random_state=random_seed).index) 
                    if sampled_demos is None:
                        sampled_demos = self.train_df_list[i].loc[ids]
                    else:
                        sampled_demos = pd.concat([sampled_demos, self.train_df_list[i].loc[ids]], ignore_index=True)
                    sampled_demos = sampled_demos.sample(n_shots, random_state=random_seed) # ensure that the texts belonging to same label are not close together
            else:
                retrieved_data = retriever.search(
                    query=query,
                    cutoff=n_shots,
                )
                ids = [item['id'] for item in retrieved_data]
                # If not enough examples are retrieved, sample randomly to fill the gap
                if len(ids) < n_shots:
                    remaining_examples = self.train_df.drop(index=ids)
                    ids.extend(remaining_examples.sample(n_shots-len(ids), random_state=random_seed).index) 
                sampled_demos = self.train_df.loc[ids]
                sampled_demos = sampled_demos.sample(n_shots, random_state=random_seed) # ensure that the texts belonging to same label are not close together
                
            # for label2 in sampled_demos['original_label']: # type: ignore
            #     category_count[label][label2] += 1
            # Prepare demonstrations for the prompt
            prompt = [
                {"role": "system", "content": system_description},
            ]
            if 'labels' not in sampled_demos.columns:
                sampled_demos['labels'] = sampled_demos['label']
            if 't-reason-deepseek' not in sampled_demos.columns:
                sampled_demos['t-reason-deepseek'] = ""
            for i, item in enumerate(zip(sampled_demos['text'], sampled_demos['label'], sampled_demos['labels'], sampled_demos['t-reason-deepseek'])): # type: ignore
                prompt.append({"role": "user", "content": item[0]})
                if 'multi-label' in self.task_type:
                    if self.reasoning:
                        content = f'''{{"reason": "{item[3]}", "is_benign": {'true' if 'benign' in item[2] else 'false'}, "is_negative": {'true' if 'negative' in item[2] else 'false'}, "is_toxic": {'true' if 'toxic' in item[2] else 'false'}, "is_spam": {'true' if 'spam' in item[2] else 'false'}}}'''
                    else:
                        content = f'''{{"is_benign": {'true' if 'benign' in item[2] else 'false'}, "is_negative": {'true' if 'negative' in item[2] else 'false'}, "is_toxic": {'true' if 'toxic' in item[2] else 'false'}, "is_spam": {'true' if 'spam' in item[2] else 'false'}}}'''
                    prompt.append({"role": "assistant", "content": content})
                else:
                    if self.reasoning:
                        content = f'''{{"reason": "{item[3]}", "label": "{item[1]}"}}'''
                    else:
                        content = item[1]
                    prompt.append({"role": "assistant", "content": content})
            prompt.append({"role": "user", "content": query})
            # prompt_json = json.dumps(prompt, ensure_ascii=False)
            # display(prompt_json)
            # break
            prompt = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True) # type: ignore
            prompts.append(TokensPrompt(prompt_token_ids=prompt)) # type: ignore
        # print(category_count) 
        return prompts

    def generate_prompts_block_variations(
            self,
            system_description: str,
            n_shots: int = 0, # the sum number of shots in the prompt
            new_shots: int = 0, 
            retrieval: Literal['random', 'lexical', 'semantic', 'balanced semantic', 'balanced lexical', 'Fine-grained balanced semantic', 'Fine-grained balanced lexical'] = 'random', # For single-task ICL, you can use 'random' or 'lexical' or'semantic' or 'balanced semantic' or 'balanced lexical', for multi-task ICL, you can use 'random' or 'lexical' or'semantic' or 'balanced semantic' or 'balanced lexical' orFine-grained balanced semantic' or 'Fine-grained balanced lexical'
            random_seed: int = 42,
            perturbation: str = '0.1',
    ):     
        prompts = [] # List of TokensPrompt
        labels = self.train_df['label'].unique()
        # Initialize the retriever based on the retrieval method if not random
        if retrieval != 'random' and n_shots > 0 and self.retriever is None and len(self.retriever_list) == 0:
            if 'balanced' in retrieval:
                if 'Fine-grained' in retrieval:
                    original_labels = self.train_df['original_label'].unique()
                    for original_label in original_labels:
                        temp_df = self.train_df[self.train_df['original_label'] == original_label]
                        temp_df.reset_index(inplace=True, drop=True)
                        self.train_df_list.append(temp_df)
                        self.retriever_list.append(self.create_retriever(retrieval.split(' ')[-1], retrieval+'_' + str(len(self.retriever_list)), temp_df))
                else:
                    for label in labels:
                        temp_df = self.train_df[self.train_df['label'] == label]
                        temp_df.reset_index(inplace=True, drop=True)
                        self.train_df_list.append(temp_df)
                        self.retriever_list.append(self.create_retriever(retrieval.split(' ')[-1], retrieval+'_' + str(len(self.retriever_list)), temp_df))
            else:
                retriever = self.create_retriever(retrieval, retrieval, self.train_df)
        # get the number of texts belonging to each label in the prompt to check the correctness of the prompt
        # category_count = {}
        # for original_label in self.train_df['label'].unique():
        #     category_count[original_label] = {}
        #     for original_label2 in self.train_df['original_label'].unique():
        #         category_count[original_label][original_label2] = 0
        for original_text,query, original_label, label, train_samples in zip(self.test_df['text'], self.test_df['test_samples_'+perturbation], self.test_df['original_label'], self.test_df['label'],self.test_df['train_samples_'+perturbation]):    
            # If retrieval method is random, sample from training data
            # demos = demos_df[demos_df['text'] == query].iloc[0][demos_column_name]
            if retrieval == 'random' or n_shots == 0:
                sampled_demos = self.train_df.sample(n_shots, random_state=random_seed)
            elif 'balanced' in retrieval:
                sampled_demos = None
                for i, retriever in enumerate(self.retriever_list):
                    retrieved_data = retriever.search(
                        query=query,
                        cutoff=int(n_shots/len(self.retriever_list)),
                    )
                    ids = [item['id'] for item in retrieved_data]
                    # If not enough examples are retrieved, sample randomly to fill the gap
                    if len(ids) < int(n_shots/len(self.retriever_list)):
                            warnings.warn(f"Not enough examples retrieved for query {query}, adding more examples by sampling from the remaining examples")
                            remaining_examples = self.train_df_list[i].drop(index=ids)
                            ids.extend(remaining_examples.sample(int(n_shots/len(self.retriever_list))-len(ids), random_state=random_seed).index) 
                    if sampled_demos is None:
                        sampled_demos = self.train_df_list[i].loc[ids]
                    else:
                        sampled_demos = pd.concat([sampled_demos, self.train_df_list[i].loc[ids]], ignore_index=True)
                    sampled_demos = sampled_demos.sample(n_shots, random_state=random_seed) # ensure that the texts belonging to same label are not close together
            else:
                retrieved_data = retriever.search(
                    query=query,
                    cutoff=n_shots,
                )
                ids = [item['id'] for item in retrieved_data]
                # If not enough examples are retrieved, sample randomly to fill the gap
                if len(ids) < n_shots:
                    remaining_examples = self.train_df.drop(index=ids)
                    ids.extend(remaining_examples.sample(n_shots-len(ids), random_state=random_seed).index) 
                sampled_demos = self.train_df.loc[ids]
                sampled_demos = sampled_demos.sample(n_shots, random_state=random_seed) # ensure that the texts belonging to same label are not close together
                
            # for label2 in sampled_demos['original_label']: # type: ignore
            #     category_count[label][label2] += 1
            # Prepare demonstrations for the prompt
            prompt = [
                {"role": "system", "content": system_description},
            ]
            # 判断是否有labels和t-reason-deepseek列
            if 'labels' not in sampled_demos.columns:
                sampled_demos['labels'] = sampled_demos['label']
            if 't-reason-deepseek' not in sampled_demos.columns:
                sampled_demos['t-reason-deepseek'] = ""
            for i, item in enumerate(zip(sampled_demos['text'], sampled_demos['label'], sampled_demos['labels'], sampled_demos['t-reason-deepseek'])): # type: ignore
                prompt.append({"role": "user", "content": item[0]})
                if 'multi-label' in self.task_type:
                    if self.reasoning:
                        content = f'''{{"reason": "{item[3]}", "is_benign": {'true' if 'benign' in item[2] else 'false'}, "is_negative": {'true' if 'negative' in item[2] else 'false'}, "is_toxic": {'true' if 'toxic' in item[2] else 'false'}, "is_spam": {'true' if 'spam' in item[2] else 'false'}}}'''
                    else:
                        content = f'''{{"is_benign": {'true' if 'benign' in item[2] else 'false'}, "is_negative": {'true' if 'negative' in item[2] else 'false'}, "is_toxic": {'true' if 'toxic' in item[2] else 'false'}, "is_spam": {'true' if 'spam' in item[2] else 'false'}}}'''
                    prompt.append({"role": "assistant", "content": content})
                else:
                    if self.reasoning:
                        content = f'''{{"reason": "{item[3]}", "label": "{item[1]}"}}'''
                    else:
                        content = item[1]
                    prompt.append({"role": "assistant", "content": content})
            if new_shots >= 0:
                prompt.append({"role": "user", "content": original_text})
                prompt.append({"role": "assistant", "content": label})
                if new_shots > 0:
                    for i in range(new_shots):
                        prompt.append({"role": "user", "content": train_samples[i]})
                        prompt.append({"role": "assistant", "content": label})
            prompt.append({"role": "user", "content": query})
            # prompt_json = json.dumps(prompt, ensure_ascii=False)
            # display(prompt_json)
            # break
            prompt = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True) # type: ignore
            prompts.append(TokensPrompt(prompt_token_ids=prompt)) # type: ignore
        # print(category_count) 
        return prompts
    def predictGuided(
            self,
            prompts: List[str],
            label_names: List[str],
            reasoning: bool
    ):
        if reasoning:
            if 'binary' in self.task_type or 'block' in self.task_type:
                guided_decoding_params = GuidedDecodingParams(regex=r'\{"reason": "[\w\s.,;:!\'()-]{0,1000}?", "label": "(benign|harmful)"\}')
            elif 'multi-class' in self.task_type:
                guided_decoding_params = GuidedDecodingParams(regex=r'\{"reason": "[\w\s.,;:!\'()-]{0,1000}?", "label": "(benign|negative|toxic|spam)"\}')
            elif 'multi-label' in self.task_type:
                guided_decoding_params = GuidedDecodingParams(regex=r'\{"reason": "[\w\s.,;:!\'()-]{0,1000}?", "is_benign": (true|false), "is_negative": (true|false), "is_toxic": (true|false), "is_spam": (true|false)\}')
        else:
            if 'binary' in self.task_type or 'block' in self.task_type:
                guided_decoding_params = GuidedDecodingParams(choice=label_names)
            elif 'multi-class' in self.task_type:
                guided_decoding_params = GuidedDecodingParams(choice=label_names)
            elif 'multi-label' in self.task_type:
                guided_decoding_params = GuidedDecodingParams(regex=r'\{"is_benign": (true|false), "is_negative": (true|false), "is_toxic": (true|false), "is_spam": (true|false)\}')
        sampling_params = SamplingParams(
            temperature=0.0,  # more deterministic 
            top_p=1.0,  # controls the cumulative probability of the top tokens to consider, set to 1 to consider all tokens
            top_k=-1,  # controls the number of top tokens to consider, set to -1 to consider all tokens
            max_tokens=3000,
            logprobs=0,
            guided_decoding=guided_decoding_params,
            stop=['}'],
            include_stop_str_in_output=True
            
        ) #, enable_prefix_caching=False, enable_chunked_prefill=False
        outputs = self.model.generate(prompts, sampling_params)
        return outputs
    def process_outputs(
            self,
            outputs,
            label_names: List[str],
    ):
        reasons = []
        predicted_label_list = []
        predicted_labels_list = []
        for output in outputs:
            text = output.outputs[0].text.replace('\n', '')
            text = text.replace('\t', '')
            if self.reasoning or self.task_type == 'multi-label':
                dict_text = json.loads(text)
                reasons.append(dict_text['reason'] if 'reason' in dict_text else '')
                if self.task_type == 'multi-label':
                    predicted_labels = []
                    for label_name in label_names:
                        if dict_text['is_' + label_name]:
                            predicted_labels.append(label_name)
                    predicted_labels_list.append(', '.join(predicted_labels))
                    predicted_label_list.append('')
                else:
                    predicted_labels_list.append('')
                    predicted_label_list.append(dict_text['label'])
            else:
                predicted_label_list.append(text)
                predicted_labels_list.append('')
                reasons.append('')
            
        return reasons, predicted_label_list, predicted_labels_list
    
    def evaluate(
            self,
            task_type: str,
            predicted_labels: List[str],
            label_names: List[str], # benign, harmful
    ):
        label_dict = {'all': ['benign', 'harmful'], 'spam': ['ham','spam'], 'toxic': ['benign', 'toxic'], 'emotion': ['positive', 'negative']}
        if task_type == 'specific_binary':
            true_labels = self.test_df['label'].tolist()
            true_labels = [label_names.index(label) for label in true_labels]
            predicted_labels = [label_names.index(label) for label in predicted_labels] # type: ignore
            cm = confusion_matrix(true_labels, predicted_labels)
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            false_positive_rate = fp / (fp + tn)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            f1_score = 2 * precision * recall / (precision + recall)
            print(f"Confusion matrix:\n{cm}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"False positive rate: {false_positive_rate}")
            print(f"Accuracy: {accuracy}")
            print(f"F1 score: {f1_score}")
            return precision, recall, false_positive_rate, accuracy, f1_score
        elif task_type == 'binary':
            precision_dict = {}
            recall_dict = {}
            false_positive_rate_dict = {}
            f1_score_dict = {}
            accuracy_dict = {}  
            self.test_df['predicted_label'] = predicted_labels
            for category in label_dict.keys():
                if category == 'all':
                    true_labels = self.test_df['label'].tolist()
                else:
                    category_df = self.test_df[self.test_df['original_label'].isin(label_dict[category])]
                    true_labels = category_df['label'].tolist()
                    predicted_labels = category_df['predicted_label'].tolist()
                true_labels = [label_names.index(label) for label in true_labels]
                predicted_labels = [label_names.index(label) for label in predicted_labels] # type: ignore
                cm = confusion_matrix(true_labels, predicted_labels)
                tn, fp, fn, tp = cm.ravel()
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                false_positive_rate = fp / (fp + tn)
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                f1_score = 2 * precision * recall / (precision + recall)
        
                precision_dict[category] = precision
                recall_dict[category] = recall
                false_positive_rate_dict[category] = false_positive_rate
                f1_score_dict[category] = f1_score
                accuracy_dict[category] = accuracy
            print(f"precision: {precision_dict}")
            print(f"recall: {recall_dict}")
            print(f"false positive rate: {false_positive_rate_dict}")
            print(f"accuracy: {accuracy_dict}")
            print(f"f1 score: {f1_score_dict}")
            return precision_dict, recall_dict, false_positive_rate_dict, accuracy_dict, f1_score_dict
        elif 'multi-class' in task_type:
            true_labels = self.test_df['label'].tolist()
            print(classification_report(true_labels, predicted_labels, labels=label_names))
            report_dict = classification_report(true_labels, predicted_labels, labels=label_names, output_dict=True)
            
            # 画confusion matrix
            cm = confusion_matrix(true_labels, predicted_labels, labels=label_names)
            ax= plt.subplot()
            #plt.xticks(rotation=70)
            #ax.set_yticks(ax.get_yticks())
            #ax.set_yticklabels(['covid', 'promo', 'financial', 'account alarm', 'insurance', 'delivery', 'prize', 'gamble', 'sex&porn', 'acquaintance'], rotation=0)
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticks(), rotation=70)
            sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")
            plt.yticks(rotation=0)

            ax.set_title('Confusion Matrix')

            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')

            ax.xaxis.set_ticklabels(label_names)
            ax.yaxis.set_ticklabels(label_names)
            plt.show()
            cm_dict = {'labels': label_names, 'cm': cm.tolist()}
            return report_dict, cm_dict
        else:
            assert False, f"Task type {task_type} is not supported"