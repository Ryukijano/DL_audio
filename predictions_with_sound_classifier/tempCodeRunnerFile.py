def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
         predictions = model(input)
         #Tensor (1,10)-> [[0.1,0.01,..,0.6]]
         predicted_index = predictions[0].argmax().item()
         predicted = class_mapping[predicted_index]
         expected = class_mapping[target]
    return predicted, expected