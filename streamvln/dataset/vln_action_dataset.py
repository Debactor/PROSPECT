import os
import torch
import json
import copy
import random
import tokenizers
import numpy as np
import transformers
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from typing import Dict, Optional, Sequence, List
from PIL import Image
from packaging import version

from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token
from llava.model import *

from streamvln.utils.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_MEMORY_TOKEN, MEMORY_TOKEN_INDEX
from streamvln.args import DataArguments

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # TODO maybe this should be changed for interleaved data?
            # if DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
            # only check for num_im=1
            num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources


def preprocess_llama_2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma(sources: List[List[Dict[str, str]]], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv: conversation_lib.Conversation = conversation_lib.default_conversation.copy()
    roles: Dict[str, str] = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations: List[str] = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source: List[Dict[str, str]] = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role: str = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids: torch.Tensor = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids: torch.Tensor = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets: torch.Tensor = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask target
    sep: str = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len: int = int(target.ne(tokenizer.pad_token_id).sum())

        rounds: List[str] = conversation.split(conv.sep)
        re_rounds = []
        for conv_idx in range(0, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))

        cur_len = 1  # Ignore <bos>
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # Re-append sep because split on this
            # Now "".join(parts)==rou

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1  # Ignore <bos>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # Ignore <bos>
            else:
                round_len = len(tokenizer(rou).input_ids) - 1  # Ignore <bos>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # Ignore <bos>

            round_len += 2  # sep: <end_of_turn>\n takes 2 tokens
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"warning: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        tokenizer.add_tokens(["<memory>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
    # stop_token_index = tokenizer.convert_tokens_to_ids("Ġstop")

    # Resolve <|im_start|> and <|im_end|> token IDs explicitly, not from list unpack
    # print(f"tokenizer.additional_special_tokens: {tokenizer.additional_special_tokens}")
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    # Fall back to first two additional_special_tokens_ids if missing
    if im_start == tokenizer.unk_token_id or im_end == tokenizer.unk_token_id:
        print(f"Tokenizer missing required special tokens: <|im_start|>, <|im_end|>. "
                           f"Current additional_special_tokens: {tokenizer.additional_special_tokens}")
        if len(tokenizer.additional_special_tokens_ids) >= 2:
            im_start, im_end = tokenizer.additional_special_tokens_ids[0], tokenizer.additional_special_tokens_ids[1]
            print(f"Using first two additional_special_tokens_ids: {im_start}, {im_end}")
        else:
            raise ValueError(f"Tokenizer missing required special tokens: <|im_start|>, <|im_end|>. "
                           f"Current additional_special_tokens: {tokenizer.additional_special_tokens}")
    
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
            if encode_id == memory_token_index:
                input_id[idx] = MEMORY_TOKEN_INDEX
            
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))  # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, "legacy", False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f"(#turns={len(re_rounds)} ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "gemma":
        return preprocess_gemma(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama_v3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

class VLNActionDataset(Dataset):
    def __init__(
        self, 
        tokenizer,
        data_args, 
        task_id
    ):
        super(VLNActionDataset, self).__init__()

        self.task_id = task_id
        self.image_size = data_args.image_size
        self.tokenizer = tokenizer
        self.transforms = data_args.transform_train
        self.image_processor = SigLipImageProcessor()

        self.num_frames = data_args.num_frames
        self.num_history = data_args.num_history
        self.num_future_steps = data_args.num_future_steps
        self.remove_init_turns = data_args.remove_init_turns

        self.video_folder = data_args.video_folder.split(',')

        self.nav_data =[]
        for vf in self.video_folder:
            anno_json = json.load(open(os.path.join(vf, 'annotations.json'), 'r'))
            for tdata in anno_json:
                tdata['video'] = os.path.join(vf, tdata['video'])
            self.nav_data += anno_json
        
        self.data_list = []
        for ep_id, item in enumerate(self.nav_data):
            instructions = item['instructions']
            actions = item['actions']
            actions_len = len(actions)
            if actions_len < 4:
                continue

            if not isinstance(instructions, list):
                instructions = [instructions]
                
            for ins_id in range(len(instructions)):
                valid_idx = 0
                if self.remove_init_turns:
                    valid_idx = self.clean_initial_rotations(instructions[ins_id], actions)
                    if valid_idx != 0:
                        invalid_len += 1

                if actions_len - valid_idx < 4:
                    continue
                
                num_rounds = (actions_len - valid_idx) // self.num_frames
                for n in range(num_rounds + 1):
                    if n * self.num_frames == actions_len - valid_idx:
                        continue
                    self.data_list.append((ep_id, ins_id, n * self.num_frames, valid_idx))

        self.idx2actions = {
            '0': 'STOP',
            '1': "↑",
            '2': "←",
            '3': "→",
        }

        self.conjunctions = [
                                'you can see ',
                                'in front of you is ',
                                'there is ',
                                'you can spot ',
                                'you are toward the ',
                                'ahead of you is ',
                                'in your sight is '
                            ]
        self.act_conjunctions = [
                                    'and then ', 
                                    'after that ', 
                                    'next ', 
                                    'the next action is ',
                                    'followed by ', 
                                    'leading to ', 
                                    'continuing ',
                                    'subsequently ', 
                                    'proceeding to '
                                ]
        
        prompt = f"You are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        answer = ""
        self.conversations = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]

    def __len__(self):
        return len(self.data_list)
    
    @property
    def task(self):
        return self.task_id
    
    def actions2text(self, actions):
        converted_sequence = []         
        for action in actions:
            act_text = self.idx2actions[str(action)]
            if type(act_text) == list:
                act_text = random.choice(act_text)
            converted_sequence.append(act_text)
        
        text = ''.join(converted_sequence)
        return text
    
    def prepare_conversation(self, conversation, actions): 
        i = 0
        sources = []
        t = 0
        while i < len(actions):
            source = copy.deepcopy(conversation)
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            step_actions = actions[i:i+self.num_future_steps]
            answer = self.actions2text(step_actions)
            if i == 0:
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."
            
            source[1]["value"] = answer
            i += len(step_actions)
            t += 1
            sources.extend(source)
        return sources
    
    def __getitem__(self, i):
        ep_id, ins_id, start_idx, valid_idx = self.data_list[i]
        data = self.nav_data[ep_id]
        video_path = data['video']
        video_frames = sorted(os.listdir(os.path.join(video_path, 'rgb')))

        instructions = data.get("instructions", None)
        if not isinstance(instructions, list):
            instructions = [instructions]

        actions = data['actions'][1+valid_idx:] + [0]
        actions_len = len(actions)
        time_ids = np.arange(start_idx, min(start_idx + self.num_frames, actions_len))
        assert len(time_ids) > 0
        actions = np.array(actions)[time_ids]

        start_idx, end_idx, interval = time_ids[0]+valid_idx, time_ids[-1]+1+valid_idx, self.num_future_steps
        sample_step_ids = np.arange(start_idx, end_idx, interval, dtype=np.int32)
        sample_frames = [os.path.join(video_path, 'rgb', video_frames[i]) for i in sample_step_ids]

        if time_ids[0] != 0:
            history_step_ids = np.arange(0+valid_idx, time_ids[0]+valid_idx, max(time_ids[0] // self.num_history, 1))
            history_frames = [os.path.join(video_path, 'rgb', video_frames[i]) for i in history_step_ids]
        else:
            history_frames = []
            
        # Load future frames (world model training)
        # Causal residual prediction: each dialogue round predicts the next observation frame
        # sample_step_ids e.g. [0, 4, 8, 12, 16, 20, 24, 28] (8 frames, 8 dialogue rounds)
        # Each round predicts 4 actions, mapped to the next observation frame (interval=4):
        #   round 0 (frame_0):  4 actions -> frame_4
        #   round 1 (frame_4):  4 actions -> frame_8
        #   ...
        #   round 7 (frame_28): 4 actions -> frame_32
        all_future_frames = []
        for sample_idx in sample_step_ids:
            # Next observation frame = current frame + interval (num_future_steps)
            next_frame_idx = sample_idx + interval
            if next_frame_idx < len(video_frames):
                next_frame = os.path.join(video_path, 'rgb', video_frames[next_frame_idx])
            else:
                # Out of range: use last frame
                next_frame = os.path.join(video_path, 'rgb', video_frames[len(video_frames) - 1])
            all_future_frames.append(next_frame)
        
        images = []
        for image_file in history_frames + sample_frames:
            image = Image.open(image_file).convert('RGB')
            if self.transforms is not None:
                image = self.transforms(image)
            
            image = self.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0] # [3, H, W]
            images.append(image)

        images = torch.stack(images)
        
        # Load future frame images (one next observation frame per dialogue round)
        all_future_images = []
        for next_frame_file in all_future_frames:
            image = Image.open(next_frame_file).convert('RGB')
            if self.transforms is not None:
                image = self.transforms(image)
            image = self.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0]  # [3, H, W]
            all_future_images.append(image)
        future_images = torch.stack(all_future_images)  # [num_rounds, 3, H, W]

        # (Optional) Load future depth maps (explicit prediction mode; ignored in latent mode)
        # Load from video_path/depth/ if present; otherwise None
        future_depths = None
        depth_dir = os.path.join(video_path, 'depth')
        if os.path.isdir(depth_dir):
            try:
                depth_frames_list = sorted(os.listdir(depth_dir))
                all_future_depths = []
                for sample_idx in sample_step_ids:
                    next_frame_idx = sample_idx + interval
                    if next_frame_idx < len(depth_frames_list):
                        depth_file = os.path.join(depth_dir, depth_frames_list[next_frame_idx])
                    else:
                        depth_file = os.path.join(depth_dir, depth_frames_list[len(depth_frames_list) - 1])
                    # Depth maps are often 16-bit PNG (Matterport3D); divide by 1000 for meters
                    depth_img = Image.open(depth_file)
                    depth_arr = np.array(depth_img, dtype=np.float32)
                    if depth_arr.max() > 100:  # 16-bit PNG (mm -> m)
                        depth_arr = depth_arr / 1000.0
                    depth_tensor = torch.from_numpy(depth_arr)  # [H, W]
                    all_future_depths.append(depth_tensor)
                future_depths = torch.stack(all_future_depths)  # [num_rounds, H, W]
            except Exception:
                future_depths = None
        
        sources = copy.deepcopy(self.conversations)

        if start_idx != 0:
            sources[0]["value"] += f' These are your historical observations: {DEFAULT_MEMORY_TOKEN}.'
        
        sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instructions[ins_id])
        interleave_sources = self.prepare_conversation(sources, list(actions))
        
        data_dict = preprocess([interleave_sources], self.tokenizer, True)

        return data_dict["input_ids"][0], \
            data_dict["labels"][0], \
            images, \
            torch.tensor(time_ids), \
            self.task, \
            future_images, \
            future_depths  # future frames + optional depth maps

def pad_tensors(tensors, lens=None, max_len=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
        if len(lens) == 1 and lens[0] == max_len:
            return tensors
    if max_len is None:
        max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].shape[1:]
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, *hid, dtype=dtype).to(tensors[0].device)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output

def collate_fn(batch, tokenizer):
    input_ids_batch, labels_batch, image_batch, time_ids_batch, task_type_batch, future_images_batch, future_depths_batch = zip(*batch)
    input_ids_batch = pad_sequence(input_ids_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_batch = pad_sequence(labels_batch, batch_first=True, padding_value=IGNORE_INDEX)
    
    input_ids_batch = input_ids_batch[:, :tokenizer.model_max_length]
    labels_batch = labels_batch[:, :tokenizer.model_max_length]
    attention_mask = input_ids_batch.ne(tokenizer.pad_token_id)
    
    img_lens = np.array([i.size(0) for i in image_batch])

    if time_ids_batch[0] is not None:
        time_ids_batch = pad_sequence(time_ids_batch, batch_first=True, padding_value=-1)
    
    image_batch = pad_tensors(image_batch, img_lens)
    
    # Pad future_images (one next observation frame per dialogue round)
    # future_images_batch: list of [num_rounds, 3, H, W]
    # Pad when batches have different num_rounds
    # QA tasks use empty tensor [0, 3, H, W], max_rounds=0
    num_rounds_list = [fi.shape[0] for fi in future_images_batch]
    max_rounds = max(num_rounds_list) if num_rounds_list else 0
    
    # Process future_images only when max_rounds > 0 (at least one navigation sample)
    if max_rounds > 0:
        # Build padded batch
        batch_size = len(future_images_batch)
        future_images_padded = torch.zeros(
            batch_size, max_rounds,
            future_images_batch[0].shape[1],  # 3
            future_images_batch[0].shape[2],  # H
            future_images_batch[0].shape[3],  # W
            dtype=future_images_batch[0].dtype
        )
        
        for i, fi in enumerate(future_images_batch):
            if fi.shape[0] > 0:  # Copy non-empty future_images (navigation tasks only)
                future_images_padded[i, :fi.shape[0]] = fi
        
        future_images_batch = future_images_padded  # [B, max_rounds, 3, H, W]
    else:
        # All QA samples: return None; forward checks future_images to enable world model
        future_images_batch = None
    
    # Pad future_depths (optional; non-None only when depth/ exists in dataset)
    # future_depths_batch: list of [num_rounds, H, W] or None
    has_any_depth = any(fd is not None for fd in future_depths_batch)
    if has_any_depth and max_rounds > 0:
        # Use first non-None sample for H, W
        sample_depth = next(fd for fd in future_depths_batch if fd is not None)
        H_d, W_d = sample_depth.shape[1], sample_depth.shape[2] if sample_depth.dim() == 3 else (sample_depth.shape[1], sample_depth.shape[2])
        # sample_depth shape is [R, H, W]
        H_d = sample_depth.shape[1]
        W_d = sample_depth.shape[2]
        batch_size_d = len(future_depths_batch)
        future_depths_padded = torch.zeros(batch_size_d, max_rounds, H_d, W_d, dtype=torch.float32)
        for i, fd in enumerate(future_depths_batch):
            if fd is not None and fd.shape[0] > 0:
                n = min(fd.shape[0], max_rounds)
                future_depths_padded[i, :n] = fd[:n]
        future_depths_batch = future_depths_padded  # [B, max_rounds, H, W]
    else:
        future_depths_batch = None

    return {'images': image_batch, 
            'time_ids': time_ids_batch, 
            'attention_mask': attention_mask, 
            'input_ids': input_ids_batch, 
            'labels': labels_batch, 
            'task_type': task_type_batch,
            'future_images': future_images_batch,   # future frame RGB
            'future_depths': future_depths_batch}   # future depth maps (optional)
