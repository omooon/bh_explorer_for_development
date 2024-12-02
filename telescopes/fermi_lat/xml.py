import xml.etree.ElementTree as ET
from gammapy.modeling import Dataset
from gammapy.modeling.models import SkyModel, PowerLawSpectralModel, GaussianSpatialModel
from fermipy import binned_analysis

def dataset_to_fermitools_xml(dataset, output_path):
    """
    DatasetオブジェクトをFermitoolsで使用できるXML形式に変換して保存する関数。

    Parameters:
    - dataset: Dataset オブジェクト（gammapyのDataset）
    - output_path: 出力先のXMLファイルパス
    """
    # XMLのルート要素を作成
    root = ET.Element("FERMITOOLS")
    
    # データセットの基本情報
    dataset_element = ET.SubElement(root, "dataset")
    dataset_element.set("livetime", str(dataset.livetime))
    dataset_element.set("obs_id", str(dataset.obs_id))
    
    # モデルの情報をXMLに変換して追加
    models_element = ET.SubElement(dataset_element, "models")
    
    # Dataset内の各モデルをXMLに変換
    for model in dataset.models:
        model_element = ET.SubElement(models_element, "model", type=model.__class__.__name__)
        model_element.set("name", model.name)

        # スペクトルモデルのパラメータを追加
        if hasattr(model, 'spectral_model'):
            spectral_model_element = ET.SubElement(model_element, "spectral_model", type=model.spectral_model.__class__.__name__)
            for param in model.spectral_model.parameters:
                param_element = ET.SubElement(spectral_model_element, "parameter")
                param_element.set("name", param.name)
                param_element.set("value", str(param.value))
                param_element.set("unit", str(param.unit))
                param_element.set("min", str(param.min))
                param_element.set("max", str(param.max))

        # 空間モデルのパラメータを追加
        if hasattr(model, 'spatial_model'):
            spatial_model_element = ET.SubElement(model_element, "spatial_model", type=model.spatial_model.__class__.__name__)
            for param in model.spatial_model.parameters:
                param_element = ET.SubElement(spatial_model_element, "parameter")
                param_element.set("name", param.name)
                param_element.set("value", str(param.value))
                param_element.set("unit", str(param.unit))
                param_element.set("min", str(param.min))
                param_element.set("max", str(param.max))

    # XMLツリーを作成し、ファイルとして保存
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    print(f"XML file saved to {output_path}")

# 使用例
if __name__ == "__main__":
    # 例として簡単なSkyModelとSpectralModelを作成し、Datasetを作成
    spectral_model = PowerLawSpectralModel(index=2.1, amplitude=1e-12)
    spatial_model = GaussianSpatialModel(sigma=0.5)
    sky_model = SkyModel(spectral_model=spectral_model, spatial_model=spatial_model, name="example_model")
    
    dataset = Dataset(models=[sky_model], livetime=1000, obs_id=12345)

    # DatasetをXML形式で保存
    dataset_to_fermitools_xml(dataset, "fermitools_dataset.xml")

