# CHANGELOG



## v0.3.3 (2024-04-07)

### Fix

* fix(loading): updated the subclassed model to be able to save and load the model with its weights ([`a65fbeb`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/a65fbebd3a10ccfb74511a243eaffe0b64737a1a))


## v0.3.2 (2024-04-01)

### Fix

* fix(loading): added get and from config methods for all layers to ensure loading and saving the model successfully ([`5fb3cde`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/5fb3cde3bbf74372633719679915908e24749fcb))


## v0.3.1 (2024-03-31)

### Fix

* fix(keras): converted all layers to keras subclassed layers and the model to a subclassed model for keras 3 integration ([`085a3a6`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/085a3a607d64bcb43e0f6c8898ca8cc8bcd64457))


## v0.3.0 (2024-03-30)

### Documentation

* docs(contributing): added additional install instructions for contributing docs ([`a40d30c`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/a40d30c8409fff573b6e02148181f1eeec176928))

### Feature

* feat(package): updated the package to support keras 3 ([`f14d978`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/f14d978f3a5a3019f1680c1cb64419c7a2875657))

### Refactor

* refactor(dependencies): added jax as backend ([`f4f18d6`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/f4f18d6e08b4c62666a25d967ead7afb24cce4b6))

* refactor(ci-cd): refactored the ci-cd pipeline to look at the previous 1 commit instead of 2 to get the prev version ([`6a9bbac`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/6a9bbac9898c256d7f8af467c723884f8ea6b063))


## v0.2.5 (2024-02-04)

### Fix

* fix(embeddings): Converted linear and periodic embedding layers to keras subclassed layers to prevent serialization errors when saving ([`6cc101c`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/6cc101c3a6102fd04b126f96fc62467354f0a0ca))


## v0.2.4 (2024-01-20)

### Fix

* fix(transoformer): fixed layernorm crashing on gpu without epsilon argument and fix transformer wasnt going passed depth level 1 ([`779c576`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/779c5768a87818a6b83d7a40d41b23fd9ab4fefd))


## v0.2.3 (2024-01-02)

### Build

* build(pipeline): removed the exit pipeline step since it didn&#39;t work ([`679cf86`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/679cf867a0b645c7e69a4b727ef07ce9ab0a69e7))

* build(pipeline): updated if statement for package version check ([`83d1e72`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/83d1e72d7dff20ebd3e24b953b7ffa982350846e))

* build(pipeline): added debugging info to version increment check ([`db40595`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/db4059533ae7194e21ec0786757902d651570f1b))

* build(pipeline): updated setting package version incremented output paramater ([`4a435da`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/4a435da69527bcadee018edd9f143135ddfb1291))

* build(pipeline): updated package version incremented check ([`53e2e78`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/53e2e78a664c6b34732f8afea1290db574d37202))

* build(pipeline): added check to see if package version was incremented and exit early if it wasn&#39;t ([`736f421`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/736f42110706246809b0e28b1727b0c9f31f4707))

### Documentation

* docs(ple): added ple target-aware embeddings ([`4b385f8`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/4b385f89aed4988a95aa114157a088ffa50f1b68))

### Fix

* fix(docs): added docstring and updated notebooks to demo latest code ([`020800c`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/020800c7cd03d935d6f08cc207b5bbce545c295c))

### Refactor

* refactor(code): removed old ft-transformer code ([`069dac0`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/069dac051e85ca8d46ca560b73fc554e081ce7cc))

* refactor(ple): moved ple bins calculations out of the model, its now passed as an argument ([`558cfb6`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/558cfb6ce98578723becd0d0545743ed0b736cf3))


## v0.2.2 (2023-12-30)

### Fix

* fix(ple): updated piecewise linear embedding layer to work with sequential input data ([`e89504e`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/e89504e3188032128eb1c3d88b993cb8b3c199e7))


## v0.2.1 (2023-12-29)

### Fix

* fix(embedding): updated the periodic numerical embedding layer to work with sequential inputs ([`42b4d8e`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/42b4d8ed85eaa0b7ec48cddac9a99f9617194111))


## v0.2.0 (2023-12-29)

### Chore

* chore(package): added pyarrow ([`c7b6a27`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/c7b6a274b2aadc07eb3c3e3992939f4519a8f0c6))

### Feature

* feat(model): updated ft-transformer model to support sequential tabular data ([`95ae501`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/95ae5010ab9b82c8528089fd968a89541179ce86))


## v0.1.0 (2023-12-24)

### Feature

* feat(model): converted ft-transformer to tensorflow functional api ([`500dae6`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/500dae68d18d405e57c464259daa372009cd40e4))


## v0.0.2 (2023-12-16)

### Build

* build(package): commit parser options ([`96713de`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/96713de09a2611bdd3181927dc6377fae9c806d3))

### Fix

* fix(package): package version ([`2c0e27c`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/2c0e27cf727b5514f973148db0650ac16d219e6c))

### Refactor

* refactor(pipeline): added back semantic-release version ([`356760a`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/356760aebd4be576569bfdafe717cbd32c06f30c))

* refactor(pipeline): added verbose to publish command ([`fc8766f`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/fc8766fd3c3a8a9cd9e5d97c7107825b2be00774))

* refactor(pipeline): removed semantic-release version ([`f9367b0`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/f9367b0057a2a116662cd445b396e8d3e81f7539))

* refactor: updated build config version for toml ([`26cf97f`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/26cf97feda3fda52ae9c21f354c1e5356c765d7a))


## v0.0.1 (2023-12-15)

### Unknown

* fix (pipeline): added GH_TOKEN to version bump ([`beebd1a`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/beebd1afbcc8d9869cde8da7689c76bdea0223d8))

* fix (pipeline): semantic-release install ([`fdef5c0`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/fdef5c0dfa6714445a5d224c959442a3a041a7ab))

* fix (pipeline): semantic-release version ([`8597ab0`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/8597ab03ac6110978f2bce3da1cbbe27efa51080))

* fix (pipeline): added build command ([`31c45c5`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/31c45c53bad5342975242a3a51db6b01f3c6f280))

* test (pipeline): updated dist path ([`1d513da`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/1d513da00d2d51de230dfe8bd5a1a1f80fa7b0f5))

* test (pipeline): showing folder contents ([`265d807`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/265d8070e74de816eb9838ef996b851db2c86f51))

* fix (pipeline): install semantic release ([`0b8a89c`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/0b8a89c9efc0b16270b4bd6eb3c1d6668f699e07))


## v0.0.0 (2023-12-09)

### Unknown

* fix (pipeline): installed sematic-release ([`28e10c9`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/28e10c95bc1bf7533f64cf9046c2b96c3e6de9ed))

* test (dummy): added dummy test ([`719cb4d`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/719cb4d07aa17b7f54752e07c7835f3ef8e57236))

* fix (pipeline): test import ([`7f4c636`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/7f4c6360bc9ea5689474c1227c9cd81092dbbd0b))

* fix (pipeline): updated package ([`e5c924d`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/e5c924db23e2f5c468b0f93a584dd3493fe16d92))

* initial package setup ([`6a0ea08`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/6a0ea08dbb9ea7a37d01a22c3c687850b2bc05d9))

* added python package structure ([`f4f5abf`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/f4f5abfac7416b87f728de678d52e75512f82a58))

* Initial commit ([`a263fc5`](https://github.com/ChristianOrr/sequential-ft-transformer/commit/a263fc532ecf9ae05534aaba4328620e739f01b5))
