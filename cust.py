# 定义客户端
import flwr as fl
from collections import OrderedDict
from typing import Dict

import torch
from flwr.common import NDArrays, Scalar
import random
from cnn import test,train,Net
from fenqu import testloader, trainloaders, valloaders


class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                 trainloader,
                 vallodaer) -> None:
        # 初始化函数，传入训练集加载器、验证集加载器
        super().__init__()

        # 将训练集加载器赋值给self.trainloader
        self.trainloader = trainloader
        # 将验证集加载器赋值给self.valloader
        self.valloader = vallodaer
        # 初始化模型，将模型赋值给self.model
        self.model = Net(num_classes=10)

    def set_parameters(self, parameters):
        """With the model parameters received from the server,
        overwrite the uninitialised model in this class with them."""

        state_keys = list(self.model.state_dict().keys())
        if len(parameters) != len(state_keys):
            # 长度不一致，回退当前权重，避免形状或条目不匹配导致报错
            params_dict = zip(state_keys, self.model.state_dict().values())
        else:
            params_dict = zip(state_keys, parameters)
        state_dict = OrderedDict({k: torch.as_tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract all model parameters and convert them to a list of
        NumPy arrays. The server doesn't work with PyTorch/TF/etc."""
        # 遍历模型的状态字典，将每个参数转换为NumPy数组
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """This method train the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server"""

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        # Define the optimizer -------------------------------------------------------------- Essentially the same as in the centralised example above
        optim = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        # do local training with a random epoch count between 1 and 5
        local_epochs = random.randint(1, 5)
        train(self.model, self.trainloader, optim, epochs=local_epochs)

        # collect local metrics
        loss_local, acc_local = test(self.model, self.valloader)
        grad_norm_sq = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm_sq += param.grad.detach().pow(2).sum().item()
        grad_norm = grad_norm_sq ** 0.5
        # use a synthetic data size for logging purposes
        data_size = random.randint(200, 1200)

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return self.get_parameters({}), data_size, {
            "loss": float(loss_local),
            "accuracy": float(acc_local),
            "data_size": data_size,
            "grad_norm": float(grad_norm),
            "local_epochs": local_epochs,
        }

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""

        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader) # <-------------------------- calls the `test` function, just what we did in the centralised setting (but this time using the client's local validation set)
        # send statistics back to the server
        return float(loss), len(self.valloader.dataset), {'accuracy': accuracy}

def get_evalulate_fn(testloader):  # 评估函数
    """This is a function that returns a function. The returned
    function (i.e. `evaluate_fn`) will be executed by the strategy
    at the end of each round to evaluate the stat of the global
    model."""
    def evaluate_fn(server_round: int, parameters, config):
        """This function is executed by the strategy it will instantiate
        a model and replace its parameters with those from the global model.
        The, the model will be evaluate on the test set (recall this is the
        whole MNIST test set)."""

        model = Net(num_classes=10)

        # set parameters to the model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # call test
        loss, accuracy = test(model, testloader) # <-------------------------- calls the `test` function, just what we did in the centralised setting
        return loss, {"accuracy": accuracy}

    return evaluate_fn


# now we can define the strategy
strategy = fl.server.strategy.FedAvg(fraction_fit=0.1, # let's sample 10% of the client each round to do local training
                                      fraction_evaluate=0.1, # after each round, let's sample 20% of the clients to asses how well the global model is doing
                                      min_available_clients=100, # total number of clients available in the experiment
                                      evaluate_fn=get_evalulate_fn(testloader)) # a callback to a function that the strategy can execute to evaluate the state of the global model on a centralised dataset

def generate_client_fn(trainloaders, valloaders):
    def client_fn(cid: str):
        """Returns a FlowerClient containing the cid-th data partition"""

        # 根据cid返回FlowerClient，包含cid-th数据分区
        return FlowerClient(trainloader=trainloaders[int(cid)],
                            vallodaer=valloaders[int(cid)])
    return client_fn

client_fn_callback = generate_client_fn(trainloaders, valloaders)
