#    This file contains code solely obtained from the Vaccine repository 
#    https://github.com/git-disl/Vaccine with minor edits

from typing import Any,Dict, Union
import torch
import torch.nn as nn

from transformers import Trainer
from transformers import logging
from transformers.utils import is_sagemaker_mp_enabled

from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.opt.modeling_opt import OPTAttention
from transformers.models.mistral.modeling_mistral import MistralAttention

logger = logging.get_logger(__name__)


def get_leaf_modules_with_grad(module):
    # # print([name for name,param  in module.named_parameters()])
    # if len(list(module.children())) == 0 and any(p.requires_grad for p in module.parameters()) and "lora_B" in module._get_name():
    #     return [module]
    # else:
    #     return [submodule for child in module.children() for submodule in get_leaf_modules_with_grad(child)]
    module_list= []
    for name, module in module.named_modules():
    #     if "lora_B" in name and "v_proj" in name and len(list(module.children())) == 0:
    #         module_list+= [module]
        if isinstance(module,LlamaAttention) or isinstance(module, OPTAttention) or isinstance(module, MistralAttention):
            module_list+= [module]
    # print(module_list)
    return module_list
            
            
class VaccineTrainer(Trainer):
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], *args, **kwargs
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            self.accelerator.backward(loss)
                # print("gere2")
            return loss 

        # if isinstance(self.optimizer,ESAM ):
        # print("calling sam")
        self.vaccine_state = {}
        self.vaccine_state ["hooks"] = []
        self.vaccine_state ["gradient"] = {}
        self.pre_first_step(model)
        step()
        self.after_first_step(model)
        model.zero_grad()
        self.pre_second_step(model)
        loss = step()
        self.after_second_step(model)
        # for param in model.parameters():
        #     if param.grad is not None:
        #         param.grad*= 1/2
        
        # else:
        #     loss = step()
        return loss.detach() / self.args.gradient_accumulation_steps

    @torch.no_grad()
    def pre_first_step(self, model ):
        def track_gradient_hook(module, grad_input, grad_output):
            # Store the gradients for the current layer
            self.vaccine_state["gradient"][module] = grad_output[0].detach().clone()/self.args.gradient_accumulation_steps
            # print(grad_output[0])
            
        def apply_backward_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_backward_hook(hook_fn)
            hooks.append(hook)  # Append the hook to the list
            
        # Call the function with the initial empty hooks list
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            self.vaccine_state["gradient"][layer] = 0
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.vaccine_state["hooks"])
            
    
    
    @torch.no_grad()
    def pre_second_step(self, model):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            perturbation = self.vaccine_state["gradient"][module]
            # print(perturbation[0,1,:])
            # # print(output.shape)
            # print(output[0,1,:])
            output[0].data =output[0] + perturbation
            # print(output.shape)
            return output
           
        
        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
        
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            # print(layer._get_name())
            # Apply hooks to all layers, including nested Sequential blocks
            apply_purturbation_hooks_recursive(layer, purturbation_hook, self.vaccine_state["hooks"])
        
    @torch.no_grad()
    def after_first_step(self, model):
        for hook in self.vaccine_state["hooks"]:
            hook.remove()
        self.vaccine_state["hooks"] = []
        
        # print(self.vaccine_state["gradient"].items())
        grad_norm = self._grad_norm(self.vaccine_state["gradient"])
        # logging.info(grad_norm)
        # logging.info("norm{}".format(grad_norm))
        for module in self.vaccine_state["gradient"]:
            # grad_norm = self._grad_norm(self.vaccine_state["gradient"][module])
            grad = self.vaccine_state["gradient"][module]
            scale = self. args. rho  / (grad_norm +1e-7) 
            e_r =  (grad)* scale
            self.vaccine_state["gradient"][module] = e_r.detach().clone()
   
    @torch.no_grad()
    def after_second_step(self, model):
        # disable hook here
        # for module in self.vaccine_state["e_r"]:
        #     module.weight.data -= self.vaccine_state["e_r"][module]
        for hook in self.vaccine_state["hooks"]:
            hook.remove()
        self.vaccine_state["hooks"] = []
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)



    @torch.no_grad()
    def _grad_norm(self,poison_grads_representation):
        norm = torch.norm(
                torch.stack([
                    #original sam 
                    ( poison_grads_representation[name] ).norm(p=2)
                    #asam 
                    # ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for name in poison_grads_representation
                ]),
                p=2
               )
        # norm = ( poison_grads_representation ).norm(p=2)
        return norm


class RandomVaccineTrainer(Trainer):
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], *args, **kwargs
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            self.accelerator.backward(loss)
                # print("gere2")
            return loss 

        self.vaccine_state = {}
        self.vaccine_state ["hooks"] = []
        self.vaccine_state ["gradient"] = {}
        self.pre_second_step(model)
        loss = step()
        self.after_second_step(model)
        # for param in model.parameters():
        #     if param.grad is not None:
        #         param.grad*= 1/2
        
        # else:
        #     loss = step()
        return loss.detach() / self.args.gradient_accumulation_steps

            
    
    
    @torch.no_grad()
    def pre_second_step(self, model):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            # print(perturbation[0,1,:])
            # # print(output.shape)
            # print(output[0,1,:])
            variance = self.args.rho
            # Generate samples from a Gaussian distribution
            gaussian_samples =  variance**(1/2) * torch.randn_like(output[0] )
            output[0].data =output[0] + gaussian_samples
            # print(output.shape)
            return output
           
        
        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
        
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            # print(layer._get_name())
            # Apply hooks to all layers, including nested Sequential blocks
            apply_purturbation_hooks_recursive(layer, purturbation_hook, self.vaccine_state["hooks"])
        
    
    @torch.no_grad()
    def after_second_step(self, model):
        # disable hook here
        # for module in self.vaccine_state["e_r"]:
        #     module.weight.data -= self.vaccine_state["e_r"][module]
        for hook in self.vaccine_state["hooks"]:
            hook.remove()
        self.vaccine_state["hooks"] = []
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)



    @torch.no_grad()
    def _grad_norm(self,poison_grads_representation):
        norm = torch.norm(
                torch.stack([
                    ( poison_grads_representation[name] ).norm(p=2)
                    for name in poison_grads_representation
                ]),
                p=2
               )
        # norm = ( poison_grads_representation ).norm(p=2)
        return norm
