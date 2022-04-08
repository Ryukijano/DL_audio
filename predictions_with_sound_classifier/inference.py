import torch
import torchaudio

from cnn import CNNnetwork
from urbansounddataset import UrbanSoundDataset
from train import AUDIO_DIR, ANNOTATIONS_FILE, BATCH_SIZE, SAMPLE_RATE, NUM_SAMPLES


class_mapping= [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dark_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]
   
def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
         predictions = model(input)
         #Tensor (1,10)-> [[0.1,0.01,..,0.6]]
         predicted_index = predictions[0].argmax().item()
         predicted = class_mapping[predicted_index]
         expected = class_mapping[target]
    return predicted, expected

if __name__ == "__main__":
    #load back the model
    cnn = CNNnetwork()
    state_dict = torch.load("cnnnet.pth")
    cnn.load_state_dict(state_dict)
    
    #load urban sound dataset
    mel_spectrogram = torchaudio.tranforms.Melspectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    
    usd = UrbanSoundDataset(ANNOTATIONS_FILE , AUDIO_DIR, mel_spectrogram, NUM_SAMPLES,"cpu")
    
    #get a sample from the urban sound dataset for inference
    input, target = usd[0][0], usd[0][1] #[batch size, num_channels, fr, time]
    input.unsqueeze_(0)
    
    #make an inference
    predicted, expected = predict(cnn, input, target, class_mapping)
    print(f"Predicted: '{predicted}', expected: '{expected}'") 