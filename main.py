from utils import load_image, FeatureExtractor

image = load_image("./data/test.jpg")
extractor = FeatureExtractor()
features = extractor.extract(image)
print(features.shape)