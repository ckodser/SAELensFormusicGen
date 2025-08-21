from functools import partial
from typing import Any, Callable, Literal, cast, Optional

import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformer import Loss, Output
from transformer_lens.utils import (
    USE_DEFAULT_VALUE,
    get_tokens_with_bos_removed,
    lm_cross_entropy_loss, Slice,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, MusicgenForConditionalGeneration, \
    AutoProcessor, MusicgenProcessor, StoppingCriteriaList

from sae_lens import logger


def load_model(
    model_class_name: str,
    model_name: str,
    device: str | torch.device | None = None,
    model_from_pretrained_kwargs: dict[str, Any] | None = None,
) -> HookedRootModule:
    model_from_pretrained_kwargs = model_from_pretrained_kwargs or {}

    if "n_devices" in model_from_pretrained_kwargs:
        n_devices = model_from_pretrained_kwargs["n_devices"]
        if n_devices > 1:
            logger.info("MODEL LOADING:")
            logger.info("Setting model device to cuda for d_devices")
            logger.info(f"Will use cuda:0 to cuda:{n_devices-1}")
            device = "cuda"
            logger.info("-------------")

    if model_class_name == "HookedTransformer":
        return HookedTransformer.from_pretrained_no_processing(
            model_name=model_name, device=device, **model_from_pretrained_kwargs
        )
    if model_class_name == "HookedMamba":
        try:
            from mamba_lens import HookedMamba
        except ImportError:  # pragma: no cover
            raise ValueError(
                "mamba-lens must be installed to work with mamba models. This can be added with `pip install sae-lens[mamba]`"
            )
        # HookedMamba has incorrect typing information, so we need to cast the type here
        return cast(
            HookedRootModule,
            HookedMamba.from_pretrained(
                model_name, device=cast(Any, device), **model_from_pretrained_kwargs
            ),
        )
    if model_class_name == "AutoModelForCausalLM":
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_from_pretrained_kwargs
        ).to(device)  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return HookedProxyLM(hf_model, tokenizer)
    if model_class_name == "MusicgenForConditionalGeneration":
        hf_model = MusicgenForConditionalGeneration.from_pretrained(
            model_name, **model_from_pretrained_kwargs,
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_name)
        hf_model.eval()
        return HookedProxyMG(hf_model, processor)

    # pragma: no cover
    raise ValueError(f"Unknown model class: {model_class_name}")


class HookedProxyLM(HookedRootModule):
    """
    A HookedRootModule that wraps a Huggingface AutoModelForCausalLM.
    """

    tokenizer: PreTrainedTokenizerBase
    model: torch.nn.Module

    def __init__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.setup()

    # copied and modified from base HookedRootModule
    def setup(self):
        self.mod_dict = {}
        self.named_modules_dict = {}
        self.hook_dict: dict[str, HookPoint] = {}
        for name, module in self.model.named_modules():
            if name == "":
                continue

            hook_point = HookPoint()
            hook_point.name = name  # type: ignore

            module.register_forward_hook(get_hook_fn(hook_point))

            self.hook_dict[name] = hook_point
            self.mod_dict[name] = hook_point
            self.named_modules_dict[name] = module

    def run_with_cache(self, *args: Any, **kwargs: Any):  # type: ignore
        if "names_filter" in kwargs:
            # hacky way to make sure that the names_filter is passed to our forward method
            kwargs["_names_filter"] = kwargs["names_filter"]
        return super().run_with_cache(*args, **kwargs)

    def forward(
        self,
        tokens: torch.Tensor,
        return_type: Literal["both", "logits"] = "logits",
        loss_per_token: bool = False,
        stop_at_layer: int | None = None,
        _names_filter: list[str] | None = None,
        **kwargs: Any,
    ) -> Output | Loss:
        # This is just what's needed for evals, not everything that HookedTransformer has
        if return_type not in (
            "both",
            "logits",
        ):
            raise NotImplementedError(
                "Only return_type supported is 'both' or 'logits' to match what's in evals.py and ActivationsStore"
            )

        stop_hooks = []
        if stop_at_layer is not None and _names_filter is not None:
            if return_type != "logits":
                raise NotImplementedError(
                    "stop_at_layer is not supported for return_type='both'"
                )
            stop_manager = StopManager(_names_filter)

            for hook_name in _names_filter:
                module = self.named_modules_dict[hook_name]
                stop_fn = stop_manager.get_stop_hook_fn(hook_name)
                stop_hooks.append(module.register_forward_hook(stop_fn))
        try:
            output = self.model(tokens)
            logits = _extract_logits_from_output(output)
        except StopForward:
            # If we stop early, we don't care about the return output
            return None  # type: ignore
        finally:
            for stop_hook in stop_hooks:
                stop_hook.remove()

        if return_type == "logits":
            return logits

        if tokens.device != logits.device:
            tokens = tokens.to(logits.device)
        loss = lm_cross_entropy_loss(logits, tokens, per_token=loss_per_token)
        return Output(logits, loss)

    def to_tokens(
        self,
        input: str | list[str],
        prepend_bos: bool | None = USE_DEFAULT_VALUE,
        padding_side: Literal["left", "right"] | None = USE_DEFAULT_VALUE,
        move_to_device: bool = True,
        truncate: bool = True,
    ) -> torch.Tensor:
        # Hackily modified version of HookedTransformer.to_tokens to work with ActivationsStore
        # Assumes that prepend_bos is always False, move_to_device is always False, and truncate is always False
        # copied from HookedTransformer.to_tokens

        if prepend_bos is not False:
            raise ValueError(
                "Only works with prepend_bos=False, to match ActivationsStore usage"
            )

        if padding_side is not None:
            raise ValueError(
                "Only works with padding_side=None, to match ActivationsStore usage"
            )

        if truncate is not False:
            raise ValueError(
                "Only works with truncate=False, to match ActivationsStore usage"
            )

        if move_to_device is not False:
            raise ValueError(
                "Only works with move_to_device=False, to match ActivationsStore usage"
            )

        tokens = self.tokenizer(
            input,
            return_tensors="pt",
            truncation=False,
            max_length=None,
        )["input_ids"]

        # We don't want to prepend bos but the tokenizer does it automatically, so we remove it manually
        if hasattr(self.tokenizer, "add_bos_token") and self.tokenizer.add_bos_token:  # type: ignore
            tokens = get_tokens_with_bos_removed(self.tokenizer, tokens)  # type: ignore
        return tokens  # type: ignore

class HookedProxyMG(HookedRootModule):

    """
    A HookedRootModule that wraps a Huggingface MusicgenForConditionalGeneration.
    """

    tokenizer: PreTrainedTokenizerBase
    model: MusicgenForConditionalGeneration

    def __init__(self, model: MusicgenForConditionalGeneration, processor: MusicgenProcessor):
        super().__init__()
        self.model = model
        self.processor = processor
        self.setup()

    # copied and modified from base HookedRootModule
    def setup(self):
        self.mod_dict = {}
        self.named_modules_dict = {}
        self.hook_dict: dict[str, HookPoint] = {}
        for name, module in self.model.named_modules():
            if name == "":
                continue

            hook_point = HookPoint()
            hook_point.name = name  # type: ignore

            module.register_forward_hook(get_hook_fn(hook_point))

            self.hook_dict[name] = hook_point
            self.mod_dict[name] = hook_point
            self.named_modules_dict[name] = module

    def get_caching_hooks(
        self,
        names_filter = None,
        incl_bwd: bool = False,
        device = None,
        remove_batch_dim: bool = False,
        cache: Optional[dict] = None,
        pos_slice = None,
    ) -> tuple[dict, list, list]:
        """Creates hooks to cache activations. Note: It does not add the hooks to the model.

        Args:
            names_filter (NamesFilter, optional): Which activations to cache. Can be a list of strings (hook names) or a filter function mapping hook names to booleans. Defaults to lambda name: True.
            incl_bwd (bool, optional): Whether to also do backwards hooks. Defaults to False.
            device (_type_, optional): The device to store on. Keeps on the same device as the layer if None.
            remove_batch_dim (bool, optional): Whether to remove the batch dimension (only works for batch_size==1). Defaults to False.
            cache (Optional[dict], optional): The cache to store activations in, a new dict is created by default. Defaults to None.

        Returns:
            cache (dict): The cache where activations will be stored.
            fwd_hooks (list): The forward hooks.
            bwd_hooks (list): The backward hooks. Empty if incl_bwd is False.
        """
        if cache is None:
            cache = {}

        pos_slice = Slice.unwrap(pos_slice)

        if names_filter is None:
            names_filter = lambda name: True
        elif isinstance(names_filter, str):
            filter_str = names_filter
            names_filter = lambda name: name == filter_str
        elif isinstance(names_filter, list):
            filter_list = names_filter
            names_filter = lambda name: name in filter_list
        elif callable(names_filter):
            names_filter = names_filter
        else:
            raise ValueError("names_filter must be a string, list of strings, or function")
        assert callable(names_filter)  # Callable[[str], bool]

        self.is_caching = True

        def save_hook(tensor: Tensor, hook: HookPoint, is_backward: bool = False):
            # for attention heads the pos dimension is the third from last
            if hook.name is None:
                raise RuntimeError("Hook should have been provided a name")

            hook_name = hook.name
            if is_backward:
                hook_name += "_grad"
            resid_stream = tensor.detach().to(device)
            if remove_batch_dim:
                resid_stream = resid_stream[0]

            if (
                hook.name.endswith("hook_q")
                or hook.name.endswith("hook_k")
                or hook.name.endswith("hook_v")
                or hook.name.endswith("hook_z")
                or hook.name.endswith("hook_result")
            ):
                pos_dim = -3
            else:
                # for all other components the pos dimension is the second from last
                # including the attn scores where the dest token is the second from last
                pos_dim = -2

            if (
                tensor.dim() >= -pos_dim
            ):  # check if the residual stream has a pos dimension before trying to slice
                resid_stream = pos_slice.apply(resid_stream, dim=pos_dim)
            import torch.nn.functional as F
            chunk = 640
            mask = self.cur_tokens["padding_mask"]
            seq_len = mask.size(1)
            n = (seq_len + chunk - 1) // chunk + 1
            pad_len = n * chunk - seq_len

            padded = F.pad(mask, (0, pad_len), value=1)  # [B, n*chunk]
            result = padded[:, ::chunk].reshape(-1).bool()

            resid_stream = resid_stream[:resid_stream.shape[0] // 2]
            resid_stream = resid_stream.reshape((-1, resid_stream.shape[-1]))[result]
            cache[hook_name] = resid_stream

        fwd_hooks = []
        bwd_hooks = []
        for name, _ in self.hook_dict.items():
            if names_filter(name):
                fwd_hooks.append((name, partial(save_hook, is_backward=False)))
                if incl_bwd:
                    bwd_hooks.append((name, partial(save_hook, is_backward=True)))

        return cache, fwd_hooks, bwd_hooks

    def run_with_cache(self, *args: Any, **kwargs: Any):  # type: ignore
        if "names_filter" in kwargs:
            # hacky way to make sure that the names_filter is passed to our forward method
            kwargs["_names_filter"] = kwargs["names_filter"]
        return super().run_with_cache(*args, **kwargs)

    def forward(
        self,
        tokens: torch.Tensor,
        return_type: Literal["both", "logits"] = "logits",
        loss_per_token: bool = False,
        stop_at_layer: int | None = None,
        _names_filter: list[str] | None = None,
        **kwargs: Any,
    ) -> Output | Loss:
        # This is just what's needed for evals, not everything that HookedTransformer has
        if return_type not in (
            "both",
            "logits",
        ):
            raise NotImplementedError(
                "Only return_type supported is 'both' or 'logits' to match what's in evals.py and ActivationsStore"
            )

        stop_hooks = []
        self.cur_tokens = tokens
        if _names_filter is not None:
            if return_type != "logits":
                raise NotImplementedError(
                    "stop_at_layer is not supported for return_type='both'"
                )
            stop_manager = StopManager(_names_filter)

            for hook_name in _names_filter:
                module = self.named_modules_dict[hook_name]
                stop_fn = stop_manager.get_stop_hook_fn(hook_name)
                stop_hooks.append(module.register_forward_hook(stop_fn))
        try:
            output = self.model.generate(
                **tokens,
                use_cache=False,
                stopping_criteria=StoppingCriteriaList([lambda *x, **y: True]),
            )
            logits = _extract_logits_from_output(output)
        except StopForward:
            # If we stop early, we don't care about the return output
            return None  # type: ignore
        finally:
            for stop_hook in stop_hooks:
                stop_hook.remove()

        if return_type == "logits":
            return logits

        if tokens.device != logits.device:
            tokens = tokens.to(logits.device)
        loss = lm_cross_entropy_loss(logits, tokens, per_token=loss_per_token)
        return Output(logits, loss)

    def to_tokens(
        self,
        input: str | list[str],
        prepend_bos: bool | None = USE_DEFAULT_VALUE,
        padding_side: Literal["left", "right"] | None = USE_DEFAULT_VALUE,
        move_to_device: bool = True,
        truncate: bool = True,
    ) -> torch.Tensor:
        # Hackily modified version of HookedTransformer.to_tokens to work with ActivationsStore
        # Assumes that prepend_bos is always False, move_to_device is always False, and truncate is always False
        # copied from HookedTransformer.to_tokens

        if prepend_bos is not False:
            raise ValueError(
                "Only works with prepend_bos=False, to match ActivationsStore usage"
            )

        if padding_side is not None:
            raise ValueError(
                "Only works with padding_side=None, to match ActivationsStore usage"
            )

        if truncate is not False:
            raise ValueError(
                "Only works with truncate=False, to match ActivationsStore usage"
            )

        if move_to_device is not False:
            raise ValueError(
                "Only works with move_to_device=False, to match ActivationsStore usage"
            )

        tokens = self.processor(
            audio=input,
            sampling_rate=32000,
            return_tensors="pt",
            truncation=False,
            max_length=None,
        )

        # # We don't want to prepend bos but the tokenizer does it automatically, so we remove it manually
        # if hasattr(self.tokenizer, "add_bos_token") and self.tokenizer.add_bos_token:  # type: ignore
        #     tokens = get_tokens_with_bos_removed(self.tokenizer, tokens)  # type: ignore
        return tokens  # type: ignore


def _extract_logits_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
        return output[0]
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    raise ValueError(f"Unknown output type: {type(output)}")


def get_hook_fn(hook_point: HookPoint):
    def hook_fn(module: Any, input: Any, output: Any) -> Any:  # noqa: ARG001
        if isinstance(output, torch.Tensor):
            return hook_point(output)
        if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            return (hook_point(output[0]), *output[1:])
        # if this isn't a tensor, just skip the hook entirely as this will break otherwise
        return output

    return hook_fn


class StopForward(Exception):
    pass


class StopManager:
    def __init__(self, hook_names: list[str]):
        self.hook_names = hook_names
        self.total_hook_names = len(set(hook_names))
        self.called_hook_names = set()

    def get_stop_hook_fn(self, hook_name: str) -> Callable[[Any, Any, Any], Any]:
        def stop_hook_fn(module: Any, input: Any, output: Any) -> Any:  # noqa: ARG001
            self.called_hook_names.add(hook_name)
            if len(self.called_hook_names) == self.total_hook_names:
                raise StopForward()
            return output

        return stop_hook_fn
