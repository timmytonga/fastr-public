import torchvision.models as models
import torch.nn as nn
from global_vars import GLUE_DATASETS, MLM_DATASETS
from models.resnet import ResNet18, ResNet34
from models.custom_models import SimpleConvNet


MODEL_TO_CHECKPOINT_NAME = {  # primarily for HuggingFace transformer --> set the default checkpoint per model
    "bert": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "bert-cased": "bert-base-cased"
}


def get_model(model_name: str,
              n_classes: int, args) -> nn.Module:
    pretrained = args.use_pretrained
    resume = args.resume
    if args.dataset == 'imagenet':
        print("Getting imagenet default model")
        if model_name not in models.__dict__:
            raise ValueError
        return models.__dict__[model_name]()
    if resume:
        raise NotImplementedError
        # model = torch.load(os.path.join(log_dir, "last_model.pth"))
        # d = train_data.input_size()[0]
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model_name == "resnet34":
        # model = torchvision.models.resnet34(pretrained=pretrained)
        # d = model.fc.in_features
        # model.fc = nn.Linear(d, n_classes)
        model = ResNet34(n_classes=n_classes)
    elif model_name == "simple":
        assert args.dataset == "mnist", "simple network for MNIST only. Need to fix input size for other datasets"
        model = SimpleConvNet()
    elif model_name == "resnet18":
        model = ResNet18(n_classes=n_classes)
    elif model_name == "bert":  # todo: refactor these NLP models into one file
        if args.dataset in GLUE_DATASETS:
            from transformers import AutoModelForSequenceClassification
            checkpoint = MODEL_TO_CHECKPOINT_NAME[model_name]
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=n_classes)
        elif args.dataset in MLM_DATASETS:
            from transformers import AutoModelForMaskedLM
            model_checkpoint = MODEL_TO_CHECKPOINT_NAME[model_name]
            model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
        else:
            raise NotImplementedError
    elif model_name == "bert-for-pretrain":
        raise NotImplementedError
    elif model_name == "distilbert":
        from transformers import AutoModelForMaskedLM
        model_checkpoint = MODEL_TO_CHECKPOINT_NAME[model_name]
        model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    else:
        raise ValueError(f"{model_name} Model not recognized.")
    return model
