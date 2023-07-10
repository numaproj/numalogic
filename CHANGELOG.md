# Changelog

## v0.5.0.post1 (2023-07-10)

 * [adebf98](https://github.com/numaproj/numalogic/commit/adebf98e9143ec569901c64f72a106575898b6ef) feat: add thread safety to local cache (#224)

### Contributors

 * Avik Basu

## v0.5.0 (2023-07-06)

 * [b21e246](https://github.com/numaproj/numalogic/commit/b21e2463fd929bdbb7aec4e50918d0b081e000b3) fix (RedisRegistry): avoid overwriting cache with the same key during load  (#223)
 * [0b0daed](https://github.com/numaproj/numalogic/commit/0b0daed7892cce7f08594a2a615acfdf66c04dd0) feat!: add dynamodb registry (#220)
 * [dacae49](https://github.com/numaproj/numalogic/commit/dacae49207ecda788e2615429ae5ecca3811ab99) fix: dataset slicing (#222)
 * [7778ae8](https://github.com/numaproj/numalogic/commit/7778ae856d37a8a6b260532ee7923651df180456) fix: update production key to latest key (#221)
 * [4831180](https://github.com/numaproj/numalogic/commit/483118095fb6e218c861a9b50e4d59b7d5ce11c0) examples: pipeline using block  (#216)
 * [203c100](https://github.com/numaproj/numalogic/commit/203c10040439c9e85a15dce8929293e51179bb23) Upgraded python support for 3.11 (#211)
 * [9ebc32f](https://github.com/numaproj/numalogic/commit/9ebc32f0d22975a897f16ddade40568610c9d266) feat!: introduce numalogic blocks (#206)
 * [4058811](https://github.com/numaproj/numalogic/commit/4058811b3ddc405f7abf45126b21459c6e04d748) fix: redis logging (#209)
 * [b93bce5](https://github.com/numaproj/numalogic/commit/b93bce59d54af1ef57c7fc00f0c1425dcf4ec910) fix: removing logger level setting (#205)
 * [6727296](https://github.com/numaproj/numalogic/commit/67272967ef406114b27fa393c5551c74ebcdd0da) fix: add caching logs (#203)
 * [4411aa7](https://github.com/numaproj/numalogic/commit/4411aa7f0f762a450a925c9546eb66be7de02a55) examples: update with new numalogic and pynumaflow (#202)
 * [2298be4](https://github.com/numaproj/numalogic/commit/2298be4bbac8f120eed654a45ee2656de402edb1) chore!: refactor preproc and postproc into transforms module (#201)
 * [9e838ef](https://github.com/numaproj/numalogic/commit/9e838ef3472544dca3f0bc2e2758c25a76bd520f) fix: Sparse AE for vanilla and conv (#199)
 * [3193159](https://github.com/numaproj/numalogic/commit/3193159eb1f81da6cc3445fe440cb4ef8962e457) fix: registry test_case (#197)
 * [efa1df7](https://github.com/numaproj/numalogic/commit/efa1df75fcf99874ad34ab22c31b6d2a1c876f59) chore!: Auto detect instance type while mlflow model save (#190)
 * [4a1effd](https://github.com/numaproj/numalogic/commit/4a1effd9178f6d5103bef2ea5e5c9f7c432d8d8c) feat: add redis caching (#179)
 * [0f7e6e0](https://github.com/numaproj/numalogic/commit/0f7e6e067a02a0c39954b649085adb027b5b16de) fix: allow import from Registryconfig with optional dependencies (#180)
 * [6c21e95](https://github.com/numaproj/numalogic/commit/6c21e952d37a8fcf7102f095b85fc193255c04d3) fix: stale check; conf lazy imports (#178)
 * [f1909a8](https://github.com/numaproj/numalogic/commit/f1909a8fb2e74d2b6ed563b6e12b7d42cadcb5a5) feat: redis registry (#170)
 * [794ddc6](https://github.com/numaproj/numalogic/commit/794ddc622589a2cf4ef1edb098e51009e23240ea) feat: local memory artifact cache (#165)
 * [03514d6](https://github.com/numaproj/numalogic/commit/03514d6a33a8b34c8a45f19de170167e31e22678) chore!: drop support for python 3.8 (#164)
 * [73bbad2](https://github.com/numaproj/numalogic/commit/73bbad2244fab04f208ff139407c22b766904ecd) feat: first benchmarking using KPI anomaly data (#163)
 * [ed40681](https://github.com/numaproj/numalogic/commit/ed406810cd4de07dc105c3f59313471cefb2058f) feat: support weight decay in optimizers (#161)
 * [cae88b3](https://github.com/numaproj/numalogic/commit/cae88b32af93fc5ed526e8058a4f73684f1dcfbf) chore!: use torch and lightning 2.0 (#159)

### Contributors

 * Avik Basu
 * Kushal Batra
 * Miroslav Boussarov
 * Tarun Chawla

## v0.4.1 (2023-06-20)

 * [79987db](https://github.com/numaproj/numalogic/commit/79987db16ce50bc0d110445b67c059f848f99da7) release: v0.4.1 (#218)

### Contributors

 * Avik Basu

## v0.4.0.post1 (2023-06-06)

 * [b134fa2](https://github.com/numaproj/numalogic/commit/b134fa27a05667c4a51fc5585d0b26618f0c0139) fix: redis logging (#209)

### Contributors

 * Kushal Batra

## v0.4.0 (2023-06-06)

 * [bd050c9](https://github.com/numaproj/numalogic/commit/bd050c9b422cc8cc17d004d03c43b21096a21d47) fix: removing logger level setting (#205)

### Contributors

 * Kushal Batra

## v0.4a1 (2023-06-02)

 * [1f5f458](https://github.com/numaproj/numalogic/commit/1f5f458dab3baf34cb43de7b95e10dc00cc88d71) fix: add caching logs (#203)
 * [ce93191](https://github.com/numaproj/numalogic/commit/ce93191dcc52964c46d1fd78ef9d8c56a646ca77) examples: update with new numalogic and pynumaflow (#202)
 * [fd169cf](https://github.com/numaproj/numalogic/commit/fd169cf2f373b99ef4e5caf78dee49d61f9ede88) chore!: refactor preproc and postproc into transforms module (#201)
 * [a2b00c1](https://github.com/numaproj/numalogic/commit/a2b00c179f46aa2c5c4c17e4e10bb4f0958fbfe7) fix: Sparse AE for vanilla and conv (#199)
 * [5e69f5f](https://github.com/numaproj/numalogic/commit/5e69f5f009ff893273378108d69f9b76917d6a91) fix: registry test_case (#197)

### Contributors

 * Avik Basu
 * Kushal Batra

## v0.4a0 (2023-05-11)

 * [75aea49](https://github.com/numaproj/numalogic/commit/75aea49dfa6d856c7430be00709fd739e734c250) chore!: Auto detect instance type while mlflow model save (#190)
 * [e884b90](https://github.com/numaproj/numalogic/commit/e884b90bf39e72ceb8b95acba59316bbf342e173) feat: add redis caching (#179)
 * [7834730](https://github.com/numaproj/numalogic/commit/7834730465e12cfa12f073209551cedff3af76c6) fix: allow import from Registryconfig with optional dependencies (#180)
 * [25a16f2](https://github.com/numaproj/numalogic/commit/25a16f200733f19a5fc377ed63eee390d261757d) fix: stale check; conf lazy imports (#178)

### Contributors

 * Avik Basu
 * Kushal Batra

## v0.4.dev5 (2023-05-09)

 * [54cffe3](https://github.com/numaproj/numalogic/commit/54cffe374b3e7fd62f91fa9d1b82a8b677837061) fix: optional dependency imports

### Contributors

 * Avik Basu

## v0.4.dev4 (2023-05-09)

 * [8086db1](https://github.com/numaproj/numalogic/commit/8086db121d437ce2bcec0ec11cc338847f197909) fix: stale check; conf lazy imports (#178)
 * [b664e49](https://github.com/numaproj/numalogic/commit/b664e49b5eaa90b43f64a4e7442b5179c7d32149) Prerelease 0.4 (#173)
 * [85fb527](https://github.com/numaproj/numalogic/commit/85fb527be68c325f8308f4d3070fe42b131962ff) chore!: use torch and lightning 2.0

### Contributors

 * Avik Basu

## v0.3.8 (2023-04-18)

 * [3160c2b](https://github.com/numaproj/numalogic/commit/3160c2b4a248f974bc6c6e4893e7c68c1fdd7890) feat: exponential moving average postprocessing (#156)
 * [9de8e4c](https://github.com/numaproj/numalogic/commit/9de8e4cfa7438047b8e7bd22c84bdcf859edc292) fix: validation loss not being logged (#155)

### Contributors

 * Avik Basu

## v0.3.7 (2023-03-27)

 * [b61ac1f](https://github.com/numaproj/numalogic/commit/b61ac1fbd639f482b7ecb661da47254695139299) fix: Tanhscaler nan output for constant feature (#153)
 * [69006eb](https://github.com/numaproj/numalogic/commit/69006ebfb396262d9e31cbc3ada2da7ed274e6a4) Update CODEOWNERS (#151)
 * [6b38465](https://github.com/numaproj/numalogic/commit/6b38465a0ce703215abcb7874211f0def10040f4) feat: more generic convolutional ae (#149)

### Contributors

 * Avik Basu
 * Vigith Maurice

## v0.3.6 (2023-03-22)

 * [cb5509a](https://github.com/numaproj/numalogic/commit/cb5509a2320026ec5bb8e9bca416859b48ab5ea2) add: anomaly sign and return labels for anomalies generated (#146)
 * [b6f63ef](https://github.com/numaproj/numalogic/commit/b6f63efc11fce355367834e707fb255549c06d39) fix: latest model calling (#145)
 * [6ee3446](https://github.com/numaproj/numalogic/commit/6ee34469e73ddd96a96e0a1c13e7fc3d48cea6d8) fix: transition (#144)

### Contributors

 * Kushal Batra

## v0.3.5 (2023-03-09)

 * [ea01b44](https://github.com/numaproj/numalogic/commit/ea01b44a3b6261d4b5ff02e4fd8e5cde53e838ba) feat: Sigmoid threshold (#141)

### Contributors

 * Avik Basu

## v0.3.4 (2023-03-03)

 * [ec8401d](https://github.com/numaproj/numalogic/commit/ec8401d3bbcb6a435af58101eccbfc489c37a4f2) feat: tanh preprocessing (#139)

### Contributors

 * Avik Basu

## v0.3.3 (2023-02-08)

 * [4eae629](https://github.com/numaproj/numalogic/commit/4eae6290bf297dcad6c58b265ef75b4112ff8ffd) fix!: consistency with threshold methods (#138)
 * [2ac1c2f](https://github.com/numaproj/numalogic/commit/2ac1c2f0564d170a52c4259aecb985c4412884d3) feat: static threshold estimator (#136)
 * [d3488c9](https://github.com/numaproj/numalogic/commit/d3488c9e0c6fa017ce965ddb78d41e3ea1209fa8) feat: initial config schema (#135)

### Contributors

 * Avik Basu

## v0.3.2 (2023-01-20)


### Contributors


## v0.3.1 (2023-01-12)

 * [2923718](https://github.com/numaproj/numalogic/commit/2923718b4bfd66ddaf74a84c2dc9b2a48a3a4a1f) fix: unbatch error on certain cases (#131)
 * [232302e](https://github.com/numaproj/numalogic/commit/232302ed134c8447c669b5f56e2684e426cb7e95) fix: pin protobuf to v3.20 for pytorch-lightning (#130)
 * [dca9a7a](https://github.com/numaproj/numalogic/commit/dca9a7a7b39ea229c5f306770967d10aa0285e35) feat!: merge to release v0.3 (#119)
 * [df20591](https://github.com/numaproj/numalogic/commit/df20591f5a3b45117525c4331e5668c4289bd118) fix: sklearn base import for scikit learn v1.2 (#112)
 * [d2c6293](https://github.com/numaproj/numalogic/commit/d2c6293f594ed67cc8a6ddabce1f8f0bdd84249f) fix: change example pipeline mlflow port (#96)
 * [f9c74c9](https://github.com/numaproj/numalogic/commit/f9c74c90c37fca24db3465c24326a639e93bc802) fix: allow only patch updates in torch version due to cuda build errors on mac (#90)
 * [0a5fcdf](https://github.com/numaproj/numalogic/commit/0a5fcdf6c1ea4732b2c6c97f57d983751cfe2956) fix_readme: mention namespace name when applying the pipeline (#88)

### Contributors

 * Avik Basu
 * Kushal Batra

## v0.3.0a1 (2022-12-22)

 * [88d26ec](https://github.com/numaproj/numalogic/commit/88d26ec000ee719f967dc90f4f78f57784b1db7e) feat!: convert AE variants to lightning modules (#110)
 * [ed94615](https://github.com/numaproj/numalogic/commit/ed9461517f68192d275d81ef38727bcbe62e887f) fix: fix and clean mlflow test cases (#109)

### Contributors

 * Avik Basu
 * Kushal Batra

## v0.3.0a0 (2022-12-08)

 * [78cf5b4](https://github.com/numaproj/numalogic/commit/78cf5b4e26ffd6dffa5e0dc364cbd51b220bd928) fix: fix pipeline for 0.3 release (#106)
 * [709553f](https://github.com/numaproj/numalogic/commit/709553f4ae879f7e786634b02867966756706174) fix: fix mlflow test cases (#98)
 * [701812e](https://github.com/numaproj/numalogic/commit/701812efed6742915d9cf57aef985127b3cc449d) feat!: disentangle threshold selection from the main model  (#89)

### Contributors

 * Avik Basu
 * Kushal Batra

## v0.3.0 (2023-01-05)

 * [dca9a7a](https://github.com/numaproj/numalogic/commit/dca9a7a7b39ea229c5f306770967d10aa0285e35) feat!: merge to release v0.3 (#119)

### Contributors

 * Avik Basu

## v0.2.10 (2023-01-06)

 * [c00bb14](https://github.com/numaproj/numalogic/commit/c00bb140dc481d90b70f98e8fbb1968db151b9ca) fix: Upgrade torch to 1.13.1 (#128)

### Contributors

 * Avik Basu

## v0.2.9 (2022-12-21)

 * [df20591](https://github.com/numaproj/numalogic/commit/df20591f5a3b45117525c4331e5668c4289bd118) fix: sklearn base import for scikit learn v1.2 (#112)

### Contributors

 * Avik Basu

## v0.2.8 (2022-11-29)

 * [7d5075d](https://github.com/numaproj/numalogic/commit/7d5075ddda5950fef772cf07ffdb8f949bed589f) remove mlflow full
 * [05d6071](https://github.com/numaproj/numalogic/commit/05d6071a1de4ca85ed8095bab7654c2e77f366f5) fix: have mlflow-server as an optional extra
 * [3c5bc83](https://github.com/numaproj/numalogic/commit/3c5bc8371a50a98a7ffc634b034692cc53ba597c) fix: lock file
 * [d2c6293](https://github.com/numaproj/numalogic/commit/d2c6293f594ed67cc8a6ddabce1f8f0bdd84249f) fix: change example pipeline mlflow port (#96)

### Contributors

 * Avik Basu
 * Kushal Batra

## v0.2.7 (2022-11-14)

 * [f9c74c9](https://github.com/numaproj/numalogic/commit/f9c74c90c37fca24db3465c24326a639e93bc802) fix: allow only patch updates in torch version due to cuda build errors on mac (#90)
 * [0a5fcdf](https://github.com/numaproj/numalogic/commit/0a5fcdf6c1ea4732b2c6c97f57d983751cfe2956) fix_readme: mention namespace name when applying the pipeline (#88)
 * [537fae5](https://github.com/numaproj/numalogic/commit/537fae56577954391d10b134cb318d388c1a7dd7) fix: AutoencoderPipeline logged loss mean (#55)
 * [22f8e5d](https://github.com/numaproj/numalogic/commit/22f8e5d79183d848f68c7cbeb2d88126fb326d03) chore!: make mlflow as an optional dependency (#47)

### Contributors

 * Avik Basu
 * Kushal Batra
 * diego-ponce

## v0.2.6 (2022-10-17)

 * [5703d1b](https://github.com/numaproj/numalogic/commit/5703d1b75d642242dc5c521a2e85e9cbb2527923) fix: update readme with optional mlflow dependency
 * [702f3b4](https://github.com/numaproj/numalogic/commit/702f3b45bea0890de2db6a2b61a44dd0a675d3be) fix: install extras in workflows
 * [8969e80](https://github.com/numaproj/numalogic/commit/8969e80e1831e0b6823a650f400d954de2e76c22) chore!: make mlflow as an optional dependency
 * [9cc97cb](https://github.com/numaproj/numalogic/commit/9cc97cbb7161fa61108288c0381a10b802114b4d) feat: resume training parameter (#40)

### Contributors

 * Avik Basu
 * Kushal Batra

## v0.2.5 (2022-09-28)


### Contributors


## v0.2.4 (2022-09-21)

 * [f0b0d3c](https://github.com/numaproj/numalogic/commit/f0b0d3c98bb12722dd9b3c5d064e3d0330008d07) fix: loading secondary artifacts (#16)
 * [0c506bd](https://github.com/numaproj/numalogic/commit/0c506bd53e00f6d0c9abfc722b9b3bb57c9cd59f) chore (#19)
 * [c845d09](https://github.com/numaproj/numalogic/commit/c845d099ca038b761e78f306b4e33f4924737197) Update README.md
 * [e8be7c5](https://github.com/numaproj/numalogic/commit/e8be7c513db2c1f6db1b074fd3832357fa113d1a) fix: pypi auto publish workflow (#18)
 * [fa22031](https://github.com/numaproj/numalogic/commit/fa220317229594bce67b37b13e94a22001b1a8e2) [Chore] Update README (#17)

### Contributors

 * Avik Basu
 * Kushal Batra
 * Saradhi Sreegiriraju
 * Vigith Maurice
 * amitkalamkar

## v0.2.3 (2022-08-16)

 * [6847211](https://github.com/numaproj/numalogic/commit/68472118ab3e64ca4f10ddc8f255e41dfcef3036) workflows: add pypi publish and auto release generation (#14)
 * [040584f](https://github.com/numaproj/numalogic/commit/040584f8bffb395504a3f0be95e5a74e5f4915e4) feat: adding feature for retaining fixed number of stale model (#13)

### Contributors

 * Avik Basu
 * Kushal Batra

## v0.2.2 (2022-08-03)

 * [a5ef072](https://github.com/numaproj/numalogic/commit/a5ef072ad6742dfcb025b43b90803c26d4915ff8) feat: Add support for storing preproc artifacts (scondary artifact) i… (#11)

### Contributors

 * Kushal Batra

## v0.2.1 (2022-07-21)


### Contributors


## v0.2.0 (2022-07-21)

 * [5314070](https://github.com/numaproj/numalogic/commit/5314070b927f5334f8022b6c5fed843dd248a1d4) feat: add transformers model (#8)

### Contributors

 * Kushal Batra

