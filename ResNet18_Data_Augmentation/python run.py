#
# ## Inference
# To run inference on a new image, use:
# ```python
# from inference import infer
# from model import get_model, set_device
# import utils
#
# config = utils.load_config('config.yaml')
# model_instance = get_model(config, num_classes)
# model_instance, device = set_device(model_instance)
# model_instance.load_state_dict(torch.load('path/to/best/checkpoint.pth')['model_state_dict'])
#
# image_path = 'path/to/image.jpg'
# transform = data_loader.get_transforms(config)[1]
# predicted_class = infer(model_instance, image_path, device, transform)
# print(f'Predicted class: {predicted_class}')
