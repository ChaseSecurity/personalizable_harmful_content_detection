import argparse
import pandas as pd
# from retriv import SparseRetriever, DenseRetriever
from vllm import LLM
import os

from ICLLeaner import InContextLearner
from taskDescription import system_descs
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# create the parser
parser = argparse.ArgumentParser()

# add the arguments
parser.add_argument("--harmful_category", type=str, help="The harmful category to predict. Spam, toxic and negative is supported. All represent all categories", required=True)
parser.add_argument("--classification", type=str, help="The classification to predict. binary, multi-class and multi-label is supported. Multi-class and multi-label can be used when harmful_category is all", required=True)
parser.add_argument("--model_path", type=str, help="The path of the model", required=True)
parser.add_argument("--retrieval", type=str, help="The retrieval method. random, lexical and semantic is supported", default="random")
parser.add_argument("--n_shots", type=int, help="The number of shots", default=48)
parser.add_argument("--tensor_parallel_size", type=int, help="The number of parallel(default=1)", default=2)
parser.add_argument("--gpu_memory_utilization", type=float, help="The gpu memory utilization(default=0.8)", default=0.85)
parser.add_argument("--predicted_text", type=str, help="The predicted text", default="This is a test")
parser.add_argument("--predicted_file", type=str, help="The predicted csv file. The csv file should have a column named text", default="")
parser.add_argument("--reasoning", type=bool, help="Whether to use reasoning", default=False)
# parse the arguments
args = parser.parse_args()
task_type = args.classification if args.harmful_category == "all" else "specific_binary"

#initialize the model
    
max_context_length = 32 * 1024
model = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len = max_context_length)

# train data paths
train_data_paths = {'spam': './datasets/spam/sscd_train.csv', 'toxic': './datasets/toxic/multilingual_toxicity_dataset/en_train.csv', 'emotion': './datasets/emotion/train.csv'}

# single task
if task_type == "specific_binary":
    # load train data
    train_data = pd.read_csv(train_data_paths[args.harmful_category], sep = '\t')
    # load test data
    if args.predicted_file == "":
        test_data = pd.DataFrame({'text': [args.predicted_text]})
    else:
        test_data = pd.read_csv(args.predicted_file)
    test_data['label'] = 'unkown'
    # initialize the InContextLearner
    icl = InContextLearner(task_type=task_type, reasoning=args.reasoning, model_name='', train_df=None, test_df=test_data, model=model, context_length=max_context_length, index_name="")
    # retrieve the demos
    if args.retrieval == 'random':
        train_data = (train_data.groupby("label", group_keys=False).apply(lambda x: x.sample(min(len(x), int(args.n_shots/2)), random_state=42))).reset_index(drop=True)

    index_name = f"{args.n_shots}"
    icl.index_name = index_name
    icl.train_df = train_data
    # generate prompts
    system_description = system_descs[task_type][args.harmful_category]['level1-chats']
    
    prompts = icl.generate_prompts(system_description=system_description,n_shots=args.n_shots, retrieval=args.retrieval)

    # predict
    outputs = icl.predictGuided(prompts=prompts, label_names=train_data['label'].unique().tolist(), reasoning=args.reasoning)
    reasons, predicted_label_list, predicted_labels_list = icl.process_outputs(outputs=outputs, label_names=train_data['label'].unique().tolist())        
    # print the result
    if args.predicted_file == "":
        print("Predicted text:", args.predicted_text)
        print("Predicted label:", predicted_label_list[0])
    else:
        test_data['predicted_label'] = predicted_label_list
        test_data.to_csv(f"{args.predicted_file}_predicted.csv", index=False)
        print(f"Predicted result saved to {args.predicted_file + '_predicted.csv'}")
elif task_type == "binary":
    # load train data
    train_data = pd.concat([pd.read_csv(train_data_paths['spam'], sep = '\t'), pd.read_csv(train_data_paths['toxic'], sep = '\t'), pd.read_csv(train_data_paths['emotion'], sep = '\t')], ignore_index=True)
    train_data["label"].apply(lambda x: "harmful" if x == 'spam' or x == 'toxic' or x == 'negative' else "benign")
    # load test data
    if args.predicted_file == "":
        test_data = pd.DataFrame({'text': [args.predicted_text]})
    else:
        test_data = pd.read_csv(args.predicted_file)
    test_data['label'] = 'unkown'
    # initialize the InContextLearner
    icl = InContextLearner(task_type=task_type, reasoning=args.reasoning, model_name='', train_df=None, test_df=test_data, model=model, context_length=max_context_length, index_name="")
    # retrieve the demos
    if args.retrieval == 'random':
        train_data = (train_data.groupby("original_label", group_keys=False).apply(lambda x: x.sample(min(len(x), int(args.n_shots/6)), random_state=42))).reset_index(drop=True)

    index_name = f"{args.n_shots}"
    icl.index_name = index_name
    icl.train_df = train_data
    # generate prompts
    system_description = system_descs[task_type]['level1-chats']
    
    prompts = icl.generate_prompts(system_description=system_description,n_shots=args.n_shots, retrieval=args.retrieval)

    # predict
    outputs = icl.predictGuided(prompts=prompts, label_names=train_data['label'].unique().tolist(), reasoning=args.reasoning)
    reasons, predicted_label_list, predicted_labels_list = icl.process_outputs(outputs=outputs, label_names=train_data['label'].unique().tolist())        
    # print the result
    if args.predicted_file == "":
        print("Predicted text:", args.predicted_text)
        print("Predicted label:", predicted_label_list[0])
    else:
        test_data['predicted_label'] = predicted_label_list
        test_data.to_csv(f"{args.predicted_file}_predicted.csv", index=False)
        print(f"Predicted result saved to {args.predicted_file + '_predicted.csv'}")

elif task_type == "multi-class":
    # load train data
    train_data = pd.concat([pd.read_csv(train_data_paths['spam'], sep = '\t'), pd.read_csv(train_data_paths['toxic'], sep = '\t'), pd.read_csv(train_data_paths['emotion'], sep = '\t')], ignore_index=True)
    train_data["label"].apply(lambda x: "benign" if x != 'spam' and x != 'toxic' and x != 'negative' else x)
    # load test data
    if args.predicted_file == "":
        test_data = pd.DataFrame({'text': [args.predicted_text]})
    else:
        test_data = pd.read_csv(args.predicted_file)
    test_data['label'] = 'unkown'
    # initialize the InContextLearner
    icl = InContextLearner(task_type=task_type, reasoning=args.reasoning, model_name='', train_df=None, test_df=test_data, model=model, context_length=max_context_length, index_name="")
    # retrieve the demos
    if args.retrieval == 'random':
        train_data = (train_data.groupby("original_label", group_keys=False).apply(lambda x: x.sample(min(len(x), int(args.n_shots/6)), random_state=42))).reset_index(drop=True)

    index_name = f"{args.n_shots}"
    icl.index_name = index_name
    icl.train_df = train_data
    # generate prompts
    system_description = system_descs[task_type]['level1-chats']

    prompts = icl.generate_prompts(system_description=system_description,n_shots=args.n_shots, retrieval=args.retrieval)

    # predict
    outputs = icl.predictGuided(prompts=prompts, label_names=train_data['label'].unique().tolist(), reasoning=args.reasoning)
    reasons, predicted_label_list, predicted_labels_list = icl.process_outputs(outputs=outputs, label_names=train_data['label'].unique().tolist())        
    # print the result
    if args.predicted_file == "":
        print("Predicted text:", args.predicted_text)
        print("Predicted label:", predicted_label_list[0])
    else:
        test_data['predicted_label'] = predicted_label_list
        test_data.to_csv(f"{args.predicted_file}_predicted.csv", index=False)
        print(f"Predicted result saved to {args.predicted_file + '_predicted.csv'}")

elif task_type == "multi-label":
    # load train data
    train_data = pd.concat([pd.read_csv(train_data_paths['spam'], sep = '\t'), pd.read_csv(train_data_paths['toxic'], sep = '\t'), pd.read_csv(train_data_paths['emotion'], sep = '\t')], ignore_index=True)
    train_data["label"].apply(lambda x: "benign" if x != 'spam' and x != 'toxic' and x != 'negative' else x)
    train_data["labels"] = train_data["label"].apply(lambda x: [x])
    # load test data
    if args.predicted_file == "":
        test_data = pd.DataFrame({'text': [args.predicted_text]})
    else:
        test_data = pd.read_csv(args.predicted_file)
    test_data['label'] = 'unkown'
    # initialize the InContextLearner
    icl = InContextLearner(task_type=task_type, reasoning=args.reasoning, model_name='', train_df=None, test_df=test_data, model=model, context_length=max_context_length, index_name="")
    # retrieve the demos
    if args.retrieval == 'random':
        train_data = (train_data.groupby("original_label", group_keys=False).apply(lambda x: x.sample(min(len(x), int(args.n_shots/6)), random_state=42))).reset_index(drop=True)

    index_name = f"{args.n_shots}"
    icl.index_name = index_name
    icl.train_df = train_data
    # generate prompts
    system_description = system_descs[task_type]['level1-chats']

    prompts = icl.generate_prompts(system_description=system_description,n_shots=args.n_shots, retrieval=args.retrieval)

    # predict
    outputs = icl.predictGuided(prompts=prompts, label_names=train_data['label'].unique().tolist(), reasoning=args.reasoning)
    reasons, predicted_label_list, predicted_labels_list = icl.process_outputs(outputs=outputs, label_names=train_data['label'].unique().tolist())        
    # print the result
    if args.predicted_file == "":
        print("Predicted text:", args.predicted_text)
        print("Predicted label:", predicted_labels_list[0])
    else:
        test_data['predicted_labels'] = predicted_labels_list
        test_data.to_csv(f"{args.predicted_file}_predicted.csv", index=False)
        print(f"Predicted result saved to {args.predicted_file + '_predicted.csv'}")
else:
    raise ValueError("Invalid task type")