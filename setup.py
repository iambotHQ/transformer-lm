from setuptools import setup, find_packages

setup(
    version="0.1.0",
    name="lm",
    packages=find_packages(),
    install_requires=[
        # only minimal inference requirements listed
        "attrs",
        "numpy",
        "sentencepiece",
        "torch",
        "tensorboardX",
    ],
    entry_points={
        "console_scripts": [
            "gpt2-sp-train = lm.data:sp_train",
            "gpt2-sp-encode = lm.data:sp_encode",
            "gpt2-tf-train = lm.gpt_2_tf.train:main",
            "gpt2 = lm.main:fire_main",
            "gpt2-gen = lm.inference:fire_gen_main",
            "gpt2-lm-web-ui = lm_web_ui.main:main",
            "gpt2-split-data = lm.split_data:fire_split_data",
            "gpt2-merge-files = lm.merge_files:fire_merge_files",
        ]
    },
)
